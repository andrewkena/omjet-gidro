"""
ppk_module.py — встраиваемый PPK-обсчёт траектории (RTKLIB demo5) для обработки эхолотных данных.

Зависимости: numpy, pandas; бинарники RTKLIB demo5 (convbin, rnx2rtkp)
    https://github.com/rtklibexplorer/RTKLIB/releases  (под Windows — готовые .exe)

Пример:
    from ppk_module import PPKProcessor, PPKConfig, attach_ppk, bottom_elevation

    ppk = PPKProcessor(rtklib_dir=r"C:\\RTKLIB\\bin", log=print)
    res = ppk.run("Rover.ubx", "Base.ubx", workdir="ppk_out",
                  config=PPKConfig(base_pos="single"))
    print(res.stats)                      # {'fix_%': 99.8, 'float_%': 0.2, ...}
    sound = attach_ppk(soundings_df, res.track, time_col="time", timebase="utc")
    sound["z_bottom"] = bottom_elevation(sound, depth_col="depth", antenna_to_transducer_m=1.20)
"""
from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pyproj import Transformer

GPS_EPOCH = pd.Timestamp("1980-01-06")
GPS_UTC_LEAP_S = 18                      # GPS − UTC с 2017-01-01
NAVSYS = {"G": 1, "S": 2, "R": 4, "E": 8, "J": 16, "C": 32}
Q_NAMES = {1: "fix", 2: "float", 3: "sbas", 4: "dgps", 5: "single", 6: "ppp"}
RINEX_EXT = (".obs", ".rnx")             # плюс *.??o — см. _is_rinex


@dataclass
class PPKConfig:
    """Настройки обработки — аналоги кнопок бота."""
    systems: str = "GREC"                 # GPS/GLO/GAL/BDS
    elmask_deg: float = 10.0              # 10° / 25°
    ar_ratio: float = 5.0                 # RATIO 3 / 5 / 10
    ar_mode: str = "continuous"           # continuous | fix-and-hold
    glo_ar: str = "on"                    # GLO BIAS: on (одинаковые приёмники) | autocal | off
    frequency: str = "l1+l2"
    # Координаты фазового центра базы:
    #   "single"      — усреднённое одиночное решение по записи базы (≈ метры абсолютной ошибки)
    #   "rinexhead"   — из заголовка RINEX (если он заполнен)
    #   (lat, lon, h) — точные координаты, h эллипсоидальная
    base_pos: Union[str, Sequence[float]] = "single"
    extra: dict = field(default_factory=dict)   # любые другие ключи конфига RTKLIB

    def to_conf(self) -> str:
        opts = {
            "pos1-posmode": "kinematic",
            "pos1-soltype": "combined",
            "pos1-frequency": self.frequency,
            "pos1-elmask": self.elmask_deg,
            "pos1-navsys": sum(NAVSYS[s] for s in self.systems.upper()),
            "pos1-sateph": "brdc",
            "pos2-armode": self.ar_mode,
            "pos2-gloarmode": self.glo_ar,
            "pos2-bdsarmode": "on",
            "pos2-arthres": self.ar_ratio,
            "misc-timeinterp": "on",       # база 1 Гц + ровер 10 Гц
            "out-solformat": "llh",
            "out-outhead": "on",
            "out-timesys": "gpst",
            "out-timeform": "tow",
            "out-height": "ellipsoidal",
        }
        if isinstance(self.base_pos, str):
            opts["ant2-postype"] = self.base_pos
        else:
            lat, lon, h = self.base_pos
            opts.update({"ant2-postype": "llh", "ant2-pos1": lat,
                         "ant2-pos2": lon, "ant2-pos3": h})
        opts.update(self.extra)
        return "\n".join(f"{k:<18}={v}" for k, v in opts.items()) + "\n"


@dataclass
class PPKResult:
    track: pd.DataFrame        # time_gpst, time_utc, lat, lon, h, Q, ns, sdn, sde, sdu, ratio ...
    stats: dict
    pos_file: Path


class PPKError(RuntimeError):
    pass


class PPKProcessor:
    def __init__(self, rtklib_dir: Optional[str] = None,
                 log: Callable[[str], None] = lambda s: None,
                 timeout_s: int = 3600):
        self.convbin = self._find_exe("convbin", rtklib_dir)
        self.rnx2rtkp = self._find_exe("rnx2rtkp", rtklib_dir)
        self.log = log
        self.timeout_s = timeout_s

    # ---------- публичный API ----------
    def run(self, rover: str, base: str, workdir: str = "ppk_out",
            config: Optional[PPKConfig] = None) -> PPKResult:
        config = config or PPKConfig()
        wd = Path(workdir)
        wd.mkdir(parents=True, exist_ok=True)

        rov_obs, rov_nav = self._to_rinex(Path(rover), wd, "rover")
        base_obs, base_nav = self._to_rinex(Path(base), wd, "base")
        navs = [n for n in (rov_nav, base_nav) if n and n.exists() and n.stat().st_size > 0]
        if not navs:
            raise PPKError("Нет навигационных данных (в UBX нужны SFRBX, либо передайте .nav)")

        conf = wd / f"{Path(rover).stem}_ppk.conf"
        conf.write_text(config.to_conf(), encoding="ascii")
        pos = wd / (Path(rover).stem + ".pos")

        self.log("PPK: расчёт (combined)…")
        self._exec([self.rnx2rtkp, "-k", str(conf), "-o", str(pos),
                    str(rov_obs), str(base_obs), *map(str, navs)])
        if not pos.exists():
            raise PPKError("rnx2rtkp не создал .pos")

        track = read_pos(pos)
        if track.empty:
            raise PPKError("Решение пустое: проверьте пересечение записей базы и ровера по времени")
        stats = solution_stats(track)
        self.log(f"PPK: готово, fix {stats['fix_%']:.1f}%, float {stats['float_%']:.1f}%")
        return PPKResult(track, stats, pos)

    # ---------- внутреннее ----------
    def _to_rinex(self, src: Path, wd: Path, tag: str):
        if not src.exists():
            raise PPKError(f"Файл не найден: {src}")
        # NAV-файл (обычный или .gz) ищем рядом с ИСХОДНЫМ путём, а не с
        # распакованной копией в wd — иначе для .gz-источника поиск .with_suffix
        # уйдёт в wd, где companion-файла нет и никогда не будет.
        nav_base = src.with_suffix("") if src.suffix.lower() == ".gz" else src
        if src.suffix.lower() == ".gz":
            self.log(f"Распаковка {src.name}…")
            src = _gunzip(src, wd)
        if is_rinex(src):
            nav = None
            for cand in (nav_base.with_suffix(".nav"), nav_base.with_suffix(nav_base.suffix[:-1] + "p")):
                if cand.exists():
                    nav = cand
                    break
                gz_cand = cand.with_suffix(cand.suffix + ".gz")
                if gz_cand.exists():
                    nav = _gunzip(gz_cand, wd)
                    break
            return src, nav
        obs, nav = wd / f"{src.stem}_{tag}.obs", wd / f"{src.stem}_{tag}.nav"
        self.log(f"Конвертация {src.name} → RINEX…")
        self._exec([self.convbin, "-r", "ubx", "-v", "3.04", "-od", "-os", "-oi", "-ot", "-ol",
                    "-o", str(obs), "-n", str(nav), str(src)])
        if not obs.exists() or obs.stat().st_size == 0:
            raise PPKError(f"{src.name}: нет сырых наблюдений (нужны UBX-RXM-RAWX)")
        return obs, nav

    def _exec(self, cmd):
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           timeout=self.timeout_s, creationflags=flags)
        if p.returncode != 0:
            raise PPKError(f"{Path(cmd[0]).name} завершился с кодом {p.returncode}:\n{p.stderr[-2000:]}")

    @staticmethod
    def _find_exe(name, directory):
        for cand in ([Path(directory) / (name + ext) for ext in (".exe", "")] if directory else []):
            if cand.exists():
                return str(cand)
        found = shutil.which(name)
        if not found:
            raise PPKError(f"Не найден {name}: укажите rtklib_dir или добавьте RTKLIB в PATH")
        return found


# ---------- функции без состояния: можно вызывать отдельно ----------
def _gunzip(path: Path, wd: Path) -> Path:
    """Распаковывает .gz во wd (снимая расширение .gz из имени) — RINEX часто
    распространяют сжатым (например, с CORS/IGS-станций); RTKLIB сам gzip не
    читает. Если распакованный файл уже есть — не распаковывает повторно."""
    out = wd / path.stem
    if not out.exists():
        wd.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "rb") as fin, open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return out


def is_rinex(p: Path) -> bool:
    if p.suffix.lower() == ".gz":
        p = Path(p.stem)  # "имя.24o.gz" -> смотрим на "имя.24o"
    s = p.suffix.lower()
    return s in RINEX_EXT or (len(s) == 4 and s[1:3].isdigit() and s[3] == "o")


def read_pos(path: Union[str, Path]) -> pd.DataFrame:
    """Читает .pos RTKLIB (out-timeform=tow, llh)."""
    cols = ["week", "tow", "lat", "lon", "h", "Q", "ns", "sdn", "sde", "sdu",
            "sdne", "sdeu", "sdun", "age", "ratio"]
    df = pd.read_csv(path, comment="%", sep=r"\s+", header=None, names=cols, engine="python")
    if df.empty:
        return df
    df["time_gpst"] = GPS_EPOCH + pd.to_timedelta(df["week"] * 604800 + df["tow"], unit="s").dt.round("1ms")
    df["time_utc"] = df["time_gpst"] - pd.Timedelta(seconds=GPS_UTC_LEAP_S)
    return df.sort_values("time_gpst").reset_index(drop=True)


def solution_stats(track: pd.DataFrame) -> dict:
    n = len(track)
    q = track["Q"].value_counts()
    pct = lambda k: float(100.0 * q.get(k, 0) / n) if n else 0.0
    return {"epochs": n, "fix_%": pct(1), "float_%": pct(2), "single_%": pct(5),
            "start_utc": track["time_utc"].iloc[0], "end_utc": track["time_utc"].iloc[-1],
            "sdu_fix_median_m": float(track.loc[track.Q == 1, "sdu"].median()) if q.get(1) else None}


def attach_ppk(soundings: pd.DataFrame, track: pd.DataFrame, time_col: str = "time",
               timebase: str = "utc", max_gap_s: float = 0.5,
               only_fix: bool = False) -> pd.DataFrame:
    """
    Интерполирует PPK-координаты на моменты замеров эхолота.
    timebase: "utc" или "gpst" — в какой шкале времени у вас метки эхолота.
    Добавляет: ppk_lat, ppk_lon, ppk_h (эллипс.), ppk_Q (худшее из двух соседних эпох), ppk_ok.
    """
    trk = track[track.Q == 1] if only_fix else track
    tcol = "time_utc" if timebase == "utc" else "time_gpst"
    t_trk = trk[tcol].values.astype("datetime64[ns]").astype(np.int64) / 1e9
    t_snd = pd.to_datetime(soundings[time_col]).values.astype("datetime64[ns]").astype(np.int64) / 1e9

    out = soundings.copy()
    for c in ("lat", "lon", "h"):
        out[f"ppk_{c}"] = np.interp(t_snd, t_trk, trk[c].values, left=np.nan, right=np.nan)

    i = np.clip(np.searchsorted(t_trk, t_snd), 1, len(t_trk) - 1)
    gap = t_trk[i] - t_trk[i - 1]
    q = trk["Q"].values
    out["ppk_Q"] = np.maximum(q[i - 1], q[i])
    out["ppk_ok"] = (gap <= max_gap_s) & (t_snd >= t_trk[0]) & (t_snd <= t_trk[-1])
    out.loc[~out["ppk_ok"], ["ppk_lat", "ppk_lon", "ppk_h"]] = np.nan
    return out


def bottom_elevation(df: pd.DataFrame, depth_col: str = "depth",
                     antenna_to_transducer_m: float = 0.0) -> pd.Series:
    """
    Отметка дна (эллипсоидальная) = h антенны − вертикаль антенна→излучатель − глубина.
    Крен/дифферент/качка не учитываются.
    """
    return df["ppk_h"] - antenna_to_transducer_m - df[depth_col]


# ---------- сведения о RINEX OBS файле (для контроля качества исходных данных) ----------
_SYS_NAMES = {"G": "GPS", "R": "GLO", "E": "GAL", "C": "BEI", "J": "QZS", "S": "SBAS", "I": "IRNSS"}
_BAND_LABELS = {
    "G": {"1": "L1", "2": "L2", "5": "L5", "6": "L6"},
    "R": {"1": "G1", "2": "G2", "3": "G3", "4": "G1a", "6": "G2a"},
    "E": {"1": "E1", "5": "E5a", "7": "E5b", "8": "E5(a+b)", "6": "E6"},
    "C": {"2": "B1I", "1": "B1C", "5": "B2a", "7": "B2b", "8": "B2(a+b)", "6": "B3"},
    "J": {"1": "L1", "2": "L2", "5": "L5", "6": "L6"},
    "S": {"1": "L1", "5": "L5"},
    "I": {"5": "L5", "9": "S"},
}


def _rinex_dt(parts):
    """parts: [год, месяц, день, час, мин, сек(.дробь)] строками — из заголовка
    (TIME OF FIRST/LAST OBS) или из строки эпохи (после '>')."""
    if len(parts) < 6:
        return None
    try:
        y, mo, d, h, mi = (int(x) for x in parts[:5])
        s = float(parts[5])
        return datetime(y, mo, d, h, mi, int(s), int(round((s % 1) * 1e6)), tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None


def rinex_obs_info(path) -> dict:
    """Сведения из заголовка и тела RINEX OBS файла для контроля качества
    исходных данных PPK (аналог карточки файла в некоторых Telegram-ботах для
    постобработки): период записи, интервал, число эпох/спецотметок,
    системы+сигналы+число спутников по каждой, координаты приёмника из
    заголовка (обычно грубое одиночное решение), версия RINEX, тип приёмника.

    Тело файла читается построчно без разбора самих измерений (только первый
    символ строки различает эпоху/спутника) — быстро даже для файлов
    в сотни МБ."""
    path = str(path)
    info = dict(path=path, size_bytes=os.path.getsize(path),
                version=None, rec_type=None, ant_type=None,
                approx_xyz=None, wgs84=None,
                t_first=None, t_last=None, interval=None,
                n_epochs=0, n_events=0, systems={})

    sys_obs_types: dict[str, list[str]] = {}
    sat_sets: dict[str, set] = {}
    cur_sys_cont = None
    header_done = False

    opener = gzip.open if path.lower().endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:
        for line in fh:
            if not header_done:
                label = line[60:].strip() if len(line) > 60 else ""
                if label == "RINEX VERSION / TYPE":
                    try:
                        info["version"] = float(line[0:9])
                    except ValueError:
                        pass
                elif label == "REC # / TYPE / VERS":
                    info["rec_type"] = line[20:40].strip() or None
                elif label == "ANT # / TYPE":
                    info["ant_type"] = line[20:40].strip() or None
                elif label == "APPROX POSITION XYZ":
                    try:
                        info["approx_xyz"] = (float(line[0:14]), float(line[14:28]),
                                              float(line[28:42]))
                    except ValueError:
                        pass
                elif label == "TIME OF FIRST OBS":
                    info["t_first"] = _rinex_dt(line[:44].split())
                elif label == "TIME OF LAST OBS":
                    info["t_last"] = _rinex_dt(line[:44].split())
                elif label == "INTERVAL":
                    try:
                        info["interval"] = float(line[0:10])
                    except ValueError:
                        pass
                elif label == "SYS / # / OBS TYPES":
                    sys_char = line[0]
                    if sys_char != " ":
                        cur_sys_cont = sys_char
                        sys_obs_types.setdefault(sys_char, [])
                    if cur_sys_cont:
                        sys_obs_types[cur_sys_cont].extend(line[7:60].split())
                elif label == "END OF HEADER":
                    header_done = True
                continue

            if line.startswith(">"):
                parts = line[1:].split()
                info["n_epochs"] += 1
                try:
                    flag = int(parts[6])
                except (ValueError, IndexError):
                    flag = 0
                if flag == 5:
                    info["n_events"] += 1
                dt = _rinex_dt(parts)
                if dt:
                    if info["t_first"] is None:
                        info["t_first"] = dt
                    info["t_last"] = dt
            elif line[:1] in _SYS_NAMES and line[1:3].strip().isdigit():
                sat_sets.setdefault(line[0], set()).add(int(line[1:3]))

    if info["interval"] is None and info["n_epochs"] > 1 and info["t_first"] and info["t_last"]:
        span = (info["t_last"] - info["t_first"]).total_seconds()
        info["interval"] = round(span / (info["n_epochs"] - 1), 3) if span > 0 else None

    for sys_char, prns in sat_sets.items():
        codes = sys_obs_types.get(sys_char, [])
        bands = sorted({c[1] for c in codes if len(c) >= 2 and c[0] in "CLDS"})
        labels = _BAND_LABELS.get(sys_char, {})
        info["systems"][sys_char] = dict(
            name=_SYS_NAMES.get(sys_char, sys_char), n_sat=len(prns),
            bands=[labels.get(b, f"?{b}") for b in bands])

    xyz = info["approx_xyz"]
    if xyz and any(abs(v) > 1.0 for v in xyz):
        lon, lat, h = Transformer.from_crs(4978, 4979, always_xy=True).transform(*xyz)
        info["wgs84"] = (lat, lon, h)

    return info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="PPK: rover + base → .pos + CSV")
    ap.add_argument("rover"); ap.add_argument("base")
    ap.add_argument("--rtklib", default=None); ap.add_argument("--out", default="ppk_out")
    ap.add_argument("--base-llh", nargs=3, type=float, default=None)
    ap.add_argument("--ratio", type=float, default=5.0)
    a = ap.parse_args()
    cfg = PPKConfig(ar_ratio=a.ratio, base_pos=tuple(a.base_llh) if a.base_llh else "single")
    r = PPKProcessor(a.rtklib, log=print).run(a.rover, a.base, a.out, cfg)
    r.track.to_csv(Path(a.out) / "track.csv", index=False)
    print(r.stats)
