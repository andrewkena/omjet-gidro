#!/usr/bin/env python3
"""
sl2sync — привязка глубин эхолота Lowrance (.sl2) к точному GNSS-треку.

Что делает:
  1. Читает .sl2 (глубина, координаты/скорость встроенного GPS Lowrance, время пинга).
  2. Читает GNSS-трек: NMEA-лог (GGA/RMC/ZDA), RTKLIB .pos (PPK) или CSV.
  3. Находит смещение времени sl2 → UTC (кросс-корреляция скорости + уточнение по траектории),
     при необходимости — дрейф часов эхолота (ppm).
  4. Оценивает остаточную задержку глубины (по согласованности встречных/пересекающихся галсов)
     или берёт её из параметра.
  5. Учитывает вынос антенна → датчик, осадку датчика, высоту антенны над водой.
  6. Фильтрует выбросы, пишет точки CSV, грид (.asc + .prj, GeoTIFF при наличии rasterio),
     отчёт PNG и summary.json.

Зависимости: numpy, scipy, pyproj; опционально matplotlib (отчёт), rasterio (GeoTIFF).
"""
import argparse
import csv
import json
import math
import os
import re
import struct
import sys
from datetime import date, datetime, timedelta, timezone

import numpy as np
from pyproj import CRS, Transformer
from scipy import signal
from scipy.interpolate import griddata
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

FT = 0.3048
KN = 0.514444
R_POLAR = 6356752.3142          # полярный радиус, используется в Mercator Lowrance
GPS_UTC_LEAP = 18               # GPST − UTC, с 2017 г.
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc).timestamp()


def log(msg):
    print(msg, flush=True)


# ============================================================================
# SL2
# ============================================================================
SL2_HDR = 144
SL2_FIELDS = [  # имя, смещение в кадре, формат
    ("frame_offset", 0, "<I"),
    ("frame_size", 28, "<H"),
    ("channel", 32, "<H"),
    ("frame_index", 36, "<I"),
    ("freq", 50, "<B"),
    ("depth_ft", 64, "<f"),
    ("keel_ft", 68, "<f"),
    ("speed_kn", 100, "<f"),
    ("temp_c", 104, "<f"),
    ("lon_enc", 108, "<i"),
    ("lat_enc", 112, "<i"),
    ("course_rad", 120, "<f"),
    ("alt_ft", 124, "<f"),
    ("heading_rad", 128, "<f"),
    ("time_ms", 140, "<I"),
]
CHANNEL_NAMES = {0: "Primary", 1: "Secondary", 2: "DownScan", 3: "SideLeft",
                 4: "SideRight", 5: "SideComposite", 9: "3D"}


def _find_frame(data, start, limit=1 << 20):
    """Ищет начало кадра: поле frame_offset равно собственной позиции в файле."""
    end = min(len(data) - SL2_HDR, start + limit)
    for p in range(start, end):
        if struct.unpack_from("<I", data, p)[0] == p and \
                struct.unpack_from("<H", data, p + 28)[0] >= SL2_HDR:
            return p
    return None


def read_sl2(path, channel=None):
    with open(path, "rb") as fh:
        data = fh.read()
    n = len(data)
    fmt = struct.unpack_from("<H", data, 0)[0]
    if fmt == 3:
        sys.exit("Это .sl3 — пока поддерживается только .sl2")
    if fmt != 2:
        log(f"  ! формат в заголовке = {fmt}, ожидался 2 (sl2). Пробую читать.")

    pos = next((s for s in (8, 10) if n >= s + SL2_HDR
                and struct.unpack_from("<I", data, s)[0] == s), None)
    if pos is None:
        pos = _find_frame(data, 0)
    cols = {name: [] for name, _, _ in SL2_FIELDS}
    resyncs = 0
    while pos is not None and pos + SL2_HDR <= n:
        fs = struct.unpack_from("<H", data, pos + 28)[0]
        if struct.unpack_from("<I", data, pos)[0] != pos or fs < SL2_HDR:
            resyncs += 1
            pos = _find_frame(data, pos + 1)
            continue
        for name, off, f in SL2_FIELDS:
            cols[name].append(struct.unpack_from(f, data, pos + off)[0])
        pos += fs
    if not cols["channel"]:
        sys.exit("В файле не найдено ни одного кадра sl2")
    a = {k: np.asarray(v) for k, v in cols.items()}

    chans, counts = np.unique(a["channel"], return_counts=True)
    summary = ", ".join(f"{CHANNEL_NAMES.get(int(c), c)}={k}" for c, k in zip(chans, counts))
    if channel is None:
        channel = next((c for c in (0, 2, 1) if c in chans), int(chans[np.argmax(counts)]))
    m = a["channel"] == channel
    if not m.any():
        sys.exit(f"Канал {channel} отсутствует. Есть: {summary}")
    s = {k: v[m] for k, v in a.items()}

    s["t_rel"] = s["time_ms"].astype(float) / 1000.0
    order = np.argsort(s["t_rel"], kind="stable")
    if np.any(np.diff(s["t_rel"]) < -1.0):
        log("  ! время в sl2 немонотонно — кадры отсортированы по времени")
    s = {k: v[order] for k, v in s.items()}

    s["lon"] = np.degrees(s["lon_enc"] / R_POLAR)
    s["lat"] = np.degrees(2 * np.arctan(np.exp(s["lat_enc"] / R_POLAR)) - np.pi / 2)
    s["has_gps"] = (s["lon_enc"] != 0) | (s["lat_enc"] != 0)
    s["depth_m"] = s["depth_ft"].astype(float) * FT
    s["speed_ms"] = s["speed_kn"].astype(float) * KN
    info = dict(frames_total=int(len(a["channel"])), channels=summary,
                channel=int(channel), pings=int(m.sum()), resyncs=resyncs,
                duration_s=float(s["t_rel"][-1] - s["t_rel"][0]))
    return s, info


# ============================================================================
# GNSS
# ============================================================================
NMEA_RE = re.compile(r"\$([A-Z]{2}(GGA|RMC|ZDA),[^*\r\n$]*)\*([0-9A-Fa-f]{2})")


def _nmea_ok(body, cs):
    c = 0
    for ch in body:
        c ^= ord(ch)
    return c == int(cs, 16)


def _dm(v, hemi):
    v = float(v)
    d = int(v // 100)
    x = d + (v - d * 100) / 60.0
    return -x if hemi in ("S", "W") else x


def _tod(s):
    return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + float(s[4:])


def _day_epoch(d):
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()


def read_nmea(path, force_date=None):
    with open(path, "r", errors="ignore") as fh:
        text = fh.read()
    recs, bad = [], 0
    first_date, first_date_tod = None, None
    for m in NMEA_RE.finditer(text):
        body, typ, cs = m.group(1), m.group(2), m.group(3)
        if not _nmea_ok(body, cs):
            bad += 1
            continue
        f = body.split(",")
        try:
            if typ == "GGA":
                if not (f[1] and f[2] and f[4] and f[6]) or int(f[6]) == 0:
                    continue
                alt = float(f[9]) if f[9] else np.nan
                recs.append((_tod(f[1]), _dm(f[2], f[3]), _dm(f[4], f[5]), alt, int(f[6])))
            elif typ == "RMC" and first_date is None and len(f) > 9 and f[9]:
                d = f[9]
                first_date = date(2000 + int(d[4:6]), int(d[2:4]), int(d[0:2]))
                first_date_tod = _tod(f[1]) if f[1] else None
            elif typ == "ZDA" and first_date is None and len(f) > 4 and f[2]:
                first_date = date(int(f[4]), int(f[3]), int(f[2]))
                first_date_tod = _tod(f[1]) if f[1] else None
        except (ValueError, IndexError):
            bad += 1
    if not recs:
        sys.exit("В NMEA-логе нет валидных GGA")
    r = np.array(recs, dtype=float)
    tod = r[:, 0]

    base = force_date or first_date
    if base is None:
        base = datetime.now(timezone.utc).date()
        log("  ! в логе нет даты (RMC/ZDA) и не задан --date — беру сегодняшнюю. "
            "На синхронизацию это не влияет, только на метки времени в CSV.")
    elif force_date is None and first_date_tod is not None and tod[0] - first_date_tod > 43200:
        base -= timedelta(days=1)       # лог начался до полуночи, а дата пришла после
    day = np.concatenate([[0], np.cumsum(np.diff(tod) < -43200)])
    t = _day_epoch(base) + day * 86400 + tod
    return dict(t=t, lat=r[:, 1], lon=r[:, 2], h=r[:, 3], q=r[:, 4].astype(int)), \
        dict(format="NMEA", epochs=len(t), bad_checksum=bad, height="MSL (GGA)")


POS_Q_TO_GGA = {1: 4, 2: 5, 3: 2, 4: 2, 5: 1, 6: 1}   # RTKLIB Q → код качества GGA


def read_pos(path):
    tsys, rows = "GPST", []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("%"):
                if "ecef" in line or "baseline" in line:
                    sys.exit("RTKLIB .pos: нужен вывод в lat/lon/height, а не ECEF/ENU")
                if "latitude(d'" in line:
                    sys.exit("RTKLIB .pos: выберите формат широты в градусах (deg), не d'\"")
                if "latitude" in line:
                    tsys = "UTC" if "UTC" in line else ("JST" if "JST" in line else "GPST")
                continue
            p = line.split()
            if len(p) < 6:
                continue
            try:
                if "/" in p[0]:
                    fmt = "%Y/%m/%d %H:%M:%S.%f" if "." in p[1] else "%Y/%m/%d %H:%M:%S"
                    t = datetime.strptime(p[0] + " " + p[1], fmt).replace(
                        tzinfo=timezone.utc).timestamp()
                else:
                    t = GPS_EPOCH + int(p[0]) * 604800 + float(p[1])
                rows.append((t, float(p[2]), float(p[3]), float(p[4]),
                             POS_Q_TO_GGA.get(int(p[5]), 1)))
            except ValueError:
                continue
    if not rows:
        sys.exit("В .pos не найдено строк с решениями")
    r = np.array(rows)
    t = r[:, 0] - {"GPST": GPS_UTC_LEAP, "JST": 9 * 3600}.get(tsys, 0)
    return dict(t=t, lat=r[:, 1], lon=r[:, 2], h=r[:, 3], q=r[:, 4].astype(int)), \
        dict(format=f"RTKLIB pos ({tsys})", epochs=len(t), height="как в .pos (обычно эллипсоид)")


def _parse_time(s):
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()


def read_csv_track(path):
    with open(path, newline="", errors="ignore") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        rd = csv.DictReader(fh, dialect=dialect)
        keys = {k.strip().lower(): k for k in rd.fieldnames}

        def pick(*names):
            return next((keys[n] for n in names if n in keys), None)
        kt = pick("time", "timestamp", "utc", "datetime", "gps_time")
        kla = pick("lat", "latitude")
        klo = pick("lon", "lng", "long", "longitude")
        kh = pick("h", "height", "alt", "altitude", "elev", "elevation")
        kq = pick("q", "quality", "fix", "fix_type")
        if not (kt and kla and klo):
            sys.exit("CSV: нужны колонки time, lat, lon (опционально h, q)")
        rows = []
        for r in rd:
            try:
                rows.append((_parse_time(r[kt]), float(r[kla]), float(r[klo]),
                             float(r[kh]) if kh and r[kh] else np.nan,
                             int(float(r[kq])) if kq and r[kq] else 4))
            except (ValueError, TypeError):
                continue
    r = np.array(rows)
    return dict(t=r[:, 0], lat=r[:, 1], lon=r[:, 2], h=r[:, 3], q=r[:, 4].astype(int)), \
        dict(format="CSV (время UTC)", epochs=len(r), height="из CSV")


def read_gnss(path, fmt="auto", force_date=None):
    if fmt == "auto":
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pos":
            fmt = "pos"
        elif ext == ".csv":
            fmt = "csv"
        else:
            with open(path, "r", errors="ignore") as fh:
                head = fh.read(65536)
            fmt = "nmea" if re.search(r"\$[A-Z]{2}(GGA|RMC)", head) else "pos"
    g, info = {"nmea": lambda: read_nmea(path, force_date),
               "pos": lambda: read_pos(path), "csv": lambda: read_csv_track(path)}[fmt]()
    order = np.argsort(g["t"], kind="stable")
    g = {k: v[order] for k, v in g.items()}
    keep = np.concatenate([[True], np.diff(g["t"]) > 1e-6]) & np.isfinite(g["lat"])
    g = {k: v[keep] for k, v in g.items()}
    info["rate_hz"] = round(float(1 / np.median(np.diff(g["t"]))), 2)
    info["fix_share"] = round(float(np.mean(g["q"] == 4)), 3)
    return g, info


# ============================================================================
# Вспомогательное
# ============================================================================
def make_proj(lon0, lat0):
    zone = int((lon0 + 180) // 6) + 1
    crs = CRS.from_epsg((32600 if lat0 >= 0 else 32700) + zone)
    fwd = Transformer.from_crs(4326, crs, always_xy=True)
    inv = Transformer.from_crs(crs, 4326, always_xy=True)
    return crs, fwd, inv


def interp_gap(tq, ts, ys, max_gap):
    """Линейная интерполяция; NaN вне диапазона и внутри разрывов > max_gap."""
    tq = np.asarray(tq, float)
    out = np.interp(tq, ts, ys)
    i = np.clip(np.searchsorted(ts, tq, side="right"), 1, len(ts) - 1)
    bad = (tq < ts[0]) | (tq > ts[-1]) | ((ts[i] - ts[i - 1]) > max_gap)
    out[bad] = np.nan
    return out


def nan_smooth(x, k):
    if k <= 1:
        return x
    w = np.ones(k)
    m = np.isfinite(x)
    s = np.convolve(np.where(m, x, 0.0), w, "same")
    c = np.convolve(m.astype(float), w, "same")
    out = s / np.maximum(c, 1e-9)
    out[~m] = np.nan
    return out


def ffill_bfill(x):
    x = x.copy()
    for arr in (x, x[::-1]):
        idx = np.where(np.isfinite(arr), np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        arr[:] = arr[idx]
    return x


def parabolic_min(x, y):
    i = int(np.nanargmin(y))
    at_edge = i == 0 or i == len(y) - 1
    if not at_edge and np.all(np.isfinite(y[i - 1:i + 2])):
        den = y[i - 1] - 2 * y[i] + y[i + 1]
        if den > 0:
            return x[i] + 0.5 * (y[i - 1] - y[i + 1]) / den * (x[1] - x[0]), y[i], False
    return x[i], y[i], at_edge


def gnss_motion(g, dt, max_gap, min_speed):
    """Скорость и курс по GNSS на равномерной сетке dt."""
    tg = np.arange(g["t"][0], g["t"][-1], dt)
    E = interp_gap(tg, g["t"], g["E"], max_gap)
    N = interp_gap(tg, g["t"], g["N"], max_gap)
    k = max(1, int(round(1.0 / dt)))
    vE = nan_smooth(np.gradient(E, dt), k)
    vN = nan_smooth(np.gradient(N, dt), k)
    speed = np.hypot(vE, vN)
    course = np.arctan2(vE, vN)
    hd = np.where(speed > min_speed, course, np.nan)
    hd = ffill_bfill(hd) if np.isfinite(hd).any() else np.zeros_like(hd)
    return dict(t=tg, speed=speed, cos=np.cos(hd), sin=np.sin(hd))


# ============================================================================
# Синхронизация времени
# ============================================================================
def coarse_offset(mot, ts, vs, dt, npeaks=5):
    """Грубое смещение: нормированная кросс-корреляция скорости (GNSS vs Lowrance)."""
    tsg = np.arange(ts[0], ts[-1], dt)
    b = interp_gap(tsg, ts, vs, 3.0)
    a = mot["speed"]
    ma, mb = np.isfinite(a), np.isfinite(b)
    a0 = np.where(ma, a - np.nanmean(a), 0.0)
    b0 = np.where(mb, b - np.nanmean(b), 0.0)
    corr = lambda x, y: signal.correlate(x, y, "full", method="fft")
    num = corr(a0, b0)
    ea = corr(a0 ** 2, mb.astype(float))
    eb = corr(ma.astype(float), b0 ** 2)
    cnt = corr(ma.astype(float), mb.astype(float))
    ncc = num / np.sqrt(np.maximum(ea * eb, 1e-12))
    ncc[cnt < max(0.3 * mb.sum(), 60 / dt)] = -np.inf
    lags = signal.correlation_lags(len(a), len(b), "full")
    # несколько лучших пиков: скорость бывает квазипериодичной, решает уточнение по траектории
    y = np.where(np.isfinite(ncc), ncc, -1.0)
    pk, _ = signal.find_peaks(y, distance=max(1, int(5.0 / dt)))
    if len(pk) == 0:
        pk = np.array([int(np.argmax(y))])
    pk = pk[np.argsort(y[pk])[::-1][:npeaks]]
    return [(float(mot["t"][0] + lags[k] * dt - tsg[0]), float(y[k])) for k in pk]


def pos_misfit(taus, ts_fix, Es, Ns, g, max_gap):
    """СКО расхождения позиций Lowrance и GNSS (без постоянного смещения) для набора сдвигов."""
    out = np.full(len(taus), np.nan)
    for j, tau in enumerate(taus):
        dE = Es - interp_gap(ts_fix + tau, g["t"], g["E"], max_gap)
        dN = Ns - interp_gap(ts_fix + tau, g["t"], g["N"], max_gap)
        m = np.isfinite(dE) & np.isfinite(dN)
        if m.sum() < 30:
            continue
        dE, dN = dE[m] - np.median(dE[m]), dN[m] - np.median(dN[m])
        out[j] = math.sqrt(np.mean(dE ** 2 + dN ** 2))
    return out


def estimate_time_model(s, g, mot, args, rep):
    fix = s["has_gps"] & np.concatenate(
        [[True], (np.diff(s["lat_enc"]) != 0) | (np.diff(s["lon_enc"]) != 0)])
    ts_fix = s["t_rel"][fix]
    Es, Ns = s["E_low"][fix], s["N_low"][fix]
    vs = s["speed_ms"][fix]
    t_mid = 0.5 * (s["t_rel"][0] + s["t_rel"][-1])
    if len(ts_fix) < 30:
        sys.exit("В sl2 слишком мало точек встроенного GPS — задайте смещение вручную (--offset)")

    if np.nanstd(vs) < 0.05:     # поле скорости пустое — считаем по координатам
        vs = np.hypot(np.gradient(Es, ts_fix), np.gradient(Ns, ts_fix))
    best = None
    for tau0, ncc in coarse_offset(mot, ts_fix, vs, args.grid_dt):
        taus = tau0 + np.arange(-3.0, 3.0001, 0.05)
        mis = pos_misfit(taus, ts_fix, Es, Ns, g, args.max_gap)
        if not np.isfinite(mis).any():
            continue
        tau1, rms1, edge = parabolic_min(taus, mis)
        if best is None or rms1 < best[3]:
            best = (tau0, ncc, tau1, rms1, edge, taus, mis)
    if best is None:
        sys.exit("Не удалось совместить треки — проверьте, что логи с одного выхода")
    tau0, ncc, tau1, rms1, edge, taus, mis = best
    log(f"  грубое смещение (корреляция скорости): {tau0:.2f} с, NCC = {ncc:.2f}")
    if ncc < 0.5:
        log("  ! слабая корреляция скорости — проверьте, что логи относятся к одному выходу")
    log(f"  уточнённое смещение (по траектории):  {tau1:.3f} с, СКО позиций {rms1:.2f} м")
    rep["offset_scan"] = (taus, mis)

    # дрейф часов: окна по 300 с с достаточным разнообразием курса
    win, wins = 300.0, []
    for w0 in np.arange(ts_fix[0], ts_fix[-1] - win / 2, win / 2):
        m = (ts_fix >= w0) & (ts_fix < w0 + win)
        if m.sum() < 60:
            continue
        dE, dN = np.diff(Es[m]), np.diff(Ns[m])
        mv = np.hypot(dE, dN) > 0.3
        if mv.sum() < 20 or abs(np.mean(np.exp(1j * np.arctan2(dE[mv], dN[mv])))) > 0.85:
            continue                  # прямой галс: сдвиг по времени ненаблюдаем
        tw = tau1 + np.arange(-1.5, 1.5001, 0.05)
        mw = pos_misfit(tw, ts_fix[m], Es[m], Ns[m], g, args.max_gap)
        if np.isfinite(mw).sum() < 10:
            continue
        x, _, e = parabolic_min(tw, mw)
        if not e:
            wins.append((ts_fix[m].mean(), x))
    a, b = tau1, 0.0
    if len(wins) >= 3:
        w = np.array(wins)
        coef = np.polyfit(w[:, 0] - t_mid, w[:, 1], 1)
        res = w[:, 1] - np.polyval(coef, w[:, 0] - t_mid)
        ok = np.abs(res) < max(0.05, 3 * np.median(np.abs(res)) * 1.4826)
        if ok.sum() >= 3:
            coef = np.polyfit(w[ok, 0] - t_mid, w[ok, 1], 1)
        span = s["t_rel"][-1] - s["t_rel"][0]
        if abs(coef[0]) * span > 0.05:
            b, a = coef[0], coef[1]
            log(f"  дрейф часов эхолота: {b * 1e6:+.1f} ppm "
                f"({b * span:+.2f} с за запись, окон: {int(ok.sum())})")
        else:
            log(f"  дрейф часов пренебрежимо мал ({coef[0] * 1e6:+.1f} ppm)")
        rep["windows"] = w
    else:
        log("  дрейф не оценивался: мало окон с поворотами (нужны записи > 10–15 мин)")
    return dict(a=float(a), b=float(b), t_mid=float(t_mid), coarse=tau0, ncc=ncc,
                rms_pos=float(rms1), edge=bool(edge))


# ============================================================================
# Позиции пингов, глубины, задержка
# ============================================================================
def ping_positions(t_pos, g, mot, lever, max_gap):
    E = interp_gap(t_pos, g["t"], g["E"], max_gap)
    N = interp_gap(t_pos, g["t"], g["N"], max_gap)
    h = interp_gap(t_pos, g["t"], g["h"], max_gap)
    c = np.interp(t_pos, mot["t"], mot["cos"])
    sn = np.interp(t_pos, mot["t"], mot["sin"])
    nrm = np.maximum(np.hypot(c, sn), 1e-9)
    c, sn = c / nrm, sn / nrm
    fwd, right = lever
    E = E + fwd * sn + right * c
    N = N + fwd * c - right * sn
    i = np.clip(np.searchsorted(g["t"], t_pos), 1, len(g["t"]) - 1)
    i -= (np.abs(g["t"][i - 1] - t_pos) < np.abs(g["t"][i] - t_pos)).astype(int)
    return E, N, h, g["q"][i], np.degrees(np.arctan2(sn, c)) % 360


def despike(d, window, abs_thr, rel_thr):
    med = median_filter(d, size=window, mode="nearest")
    return np.abs(d - med) <= np.maximum(abs_thr, rel_thr * med)


def cell_variance(E, N, d, cell):
    ix = np.floor((E - E.min()) / cell).astype(np.int64)
    iy = np.floor((N - N.min()) / cell).astype(np.int64)
    _, inv = np.unique(ix * 10_000_019 + iy, return_inverse=True)
    cnt = np.bincount(inv)
    s1 = np.bincount(inv, d)
    s2 = np.bincount(inv, d * d)
    m = cnt >= 2
    return float(np.sum(s2[m] - s1[m] ** 2 / cnt[m]) / max(cnt[m].sum(), 1))


def scan_latency(t_abs, d, g, mot, args):
    Ls = np.arange(-args.latency_range, args.latency_range + 1e-9, 0.05)
    vals = np.full(len(Ls), np.nan)
    for j, L in enumerate(Ls):
        E, N, *_ = ping_positions(t_abs - L, g, mot, args.lever, args.max_gap)
        m = np.isfinite(E)
        if m.sum() > 100:
            vals[j] = cell_variance(E[m], N[m], d[m], args.cell)
    L, v, edge = parabolic_min(Ls, vals)
    contrast = (np.nanmax(vals) - v) / max(v, 1e-9)
    return float(L), (Ls, vals), float(contrast), edge


# ============================================================================
# Вывод
# ============================================================================
def write_points(path, cols):
    names = list(cols)
    n = len(cols[names[0]])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        for i in range(n):
            w.writerow([cols[k][i] for k in names])


def make_grid(E, N, V, cell, max_interp, crs, base):
    x0, y0 = math.floor(E.min() / cell) * cell, math.floor(N.min() / cell) * cell
    nx = int(math.ceil((E.max() - x0) / cell)) + 1
    ny = int(math.ceil((N.max() - y0) / cell)) + 1
    if nx * ny > 25_000_000:
        sys.exit(f"Грид {nx}x{ny} слишком большой — увеличьте --grid-cell")
    ix = ((E - x0) / cell).astype(int)
    iy = ((N - y0) / cell).astype(int)
    _, inv = np.unique(ix * 10_000_019 + iy, return_inverse=True)
    cnt = np.bincount(inv)
    Ec, Nc, Vc = (np.bincount(inv, x) / cnt for x in (E, N, V))
    xs = x0 + (np.arange(nx) + 0.5) * cell
    ys = y0 + (np.arange(ny)[::-1] + 0.5) * cell          # строки сверху вниз
    GX, GY = np.meshgrid(xs, ys)
    Z = griddata((Ec, Nc), Vc, (GX, GY), method="linear")
    dist, _ = cKDTree(np.c_[Ec, Nc]).query(np.c_[GX.ravel(), GY.ravel()])
    Z[dist.reshape(Z.shape) > max_interp] = np.nan

    with open(base + ".asc", "w") as fh:
        fh.write(f"ncols {nx}\nnrows {ny}\nxllcorner {x0:.3f}\nyllcorner {y0:.3f}\n"
                 f"cellsize {cell}\nNODATA_value -9999\n")
        np.savetxt(fh, np.where(np.isfinite(Z), Z, -9999), fmt="%.3f")
    with open(base + ".prj", "w") as fh:
        fh.write(crs.to_wkt("WKT1_ESRI"))
    files = [base + ".asc"]
    try:
        import rasterio
        from rasterio.transform import from_origin
        with rasterio.open(base + ".tif", "w", driver="GTiff", height=ny, width=nx, count=1,
                           dtype="float32", crs=crs.to_wkt(), nodata=-9999,
                           transform=from_origin(x0, y0 + ny * cell, cell, cell),
                           compress="deflate") as dst:
            dst.write(np.where(np.isfinite(Z), Z, -9999).astype("float32"), 1)
        files.append(base + ".tif")
    except ImportError:
        pass
    return files, (xs, ys, Z)


def make_report(path, rep, res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0, 0]
    a.plot(rep["mot_t"] - rep["mot_t"][0], rep["mot_speed"], lw=0.8, label="GNSS")
    a.plot(rep["son_t"] - rep["mot_t"][0], rep["son_speed"], lw=0.8, alpha=0.8,
           label="Lowrance (после сдвига)")
    a.set(title="Скорость после синхронизации", xlabel="с", ylabel="м/с")
    a.legend()
    a = ax[0, 1]
    taus, mis = rep["offset_scan"]
    a.plot(taus - res["offset_s"], mis, ".-")
    a.axvline(0, color="r", ls="--")
    a.set(title=f"Подбор смещения (итог {res['offset_s']:.3f} с)",
          xlabel="сдвиг относительно итога, с", ylabel="СКО позиций, м")
    if "windows" in rep:
        a2 = a.inset_axes([0.55, 0.55, 0.42, 0.4])
        w = rep["windows"]
        a2.plot((w[:, 0] - w[0, 0]) / 60, w[:, 1] - res["offset_s"], "o")
        a2.set_title("смещение по окнам, мин → с", fontsize=8)
        a2.tick_params(labelsize=7)
    a = ax[1, 0]
    if "latency_scan" in rep:
        Ls, v = rep["latency_scan"]
        a.plot(Ls, v, ".-")
        a.axvline(res["latency_s"], color="r", ls="--")
        a.set(title="Подбор задержки глубины", xlabel="задержка, с",
              ylabel="дисперсия в ячейках, м²")
    else:
        a.text(0.5, 0.5, f"задержка задана вручную: {res['latency_s']} с",
               ha="center", transform=a.transAxes)
        a.set_axis_off()
    a = ax[1, 1]
    sc = a.scatter(rep["E"], rep["N"], c=rep["V"], s=1, cmap="viridis_r")
    a.set_aspect("equal")
    a.set(title=f"Точки ({rep['vname']})", xlabel="E, м", ylabel="N, м")
    fig.colorbar(sc, ax=a, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


# ============================================================================
# main
# ============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Привязка глубин Lowrance .sl2 к точному GNSS-треку",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("sl2", help="файл эхолота .sl2")
    p.add_argument("gnss", help="GNSS-трек: NMEA-лог, RTKLIB .pos или CSV")
    p.add_argument("-o", "--out", help="папка результатов (по умолчанию <имя_sl2>_out)")
    p.add_argument("--gnss-format", default="auto", choices=["auto", "nmea", "pos", "csv"])
    p.add_argument("--date", help="дата UTC для NMEA без RMC/ZDA, ГГГГ-ММ-ДД")
    p.add_argument("--channel", type=int, help="канал sl2 (0 Primary, 1 Secondary, 2 DownScan)")
    g_ = p.add_argument_group("синхронизация")
    g_.add_argument("--offset", type=float,
                    help="смещение времени sl2→UTC, с (пропускает авто-поиск)")
    g_.add_argument("--drift-ppm", type=float, default=0.0, help="дрейф часов при ручном --offset")
    g_.add_argument("--latency", type=float, default=0.0,
                    help="доп. задержка глубины, с (глубина относится к позиции на L с раньше)")
    g_.add_argument("--auto-latency", action="store_true",
                    help="подобрать задержку по пересечениям/встречным галсам")
    g_.add_argument("--latency-range", type=float, default=3.0, help="диапазон поиска ±, с")
    g_.add_argument("--cell", type=float, default=1.0, help="ячейка для подбора задержки, м")
    geo = p.add_argument_group("геометрия")
    geo.add_argument("--lever", type=float, nargs=2, default=[0.0, 0.0],
                     metavar=("ВПЕРЁД", "ВПРАВО"),
                     help="положение датчика относительно GNSS-антенны, м")
    geo.add_argument("--draft", type=float, default=0.0,
                     help="заглубление датчика под поверхность воды, м")
    geo.add_argument("--antenna-height", type=float,
                     help="высота фазового центра антенны над водой, м (включает отметки дна)")
    flt = p.add_argument_group("фильтрация и грид")
    flt.add_argument("--min-fix", default="any", choices=["any", "float", "fix"],
                     help="минимальное качество GNSS для точки")
    flt.add_argument("--min-depth", type=float, default=0.2)
    flt.add_argument("--max-depth", type=float, default=150.0)
    flt.add_argument("--despike-window", type=int, default=9)
    flt.add_argument("--despike-abs", type=float, default=0.3)
    flt.add_argument("--despike-rel", type=float, default=0.05)
    flt.add_argument("--max-gap", type=float, default=1.5, help="макс. разрыв GNSS-трека, с")
    flt.add_argument("--grid-cell", type=float, default=0.5)
    flt.add_argument("--max-interp", type=float, default=5.0,
                     help="не интерполировать дальше этого расстояния от данных, м")
    flt.add_argument("--no-grid", action="store_true")
    flt.add_argument("--grid-dt", type=float, default=0.1, help=argparse.SUPPRESS)
    args = p.parse_args()

    out = args.out or os.path.splitext(args.sl2)[0] + "_out"
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, os.path.splitext(os.path.basename(args.sl2))[0])
    rep, res = {}, {}

    log(f"[1/6] Читаю {args.sl2}")
    s, si = read_sl2(args.sl2, args.channel)
    log(f"  кадров {si['frames_total']} ({si['channels']}), канал {si['channel']}: "
        f"{si['pings']} пингов, {si['duration_s'] / 60:.1f} мин"
        + (f", пересинхронизаций {si['resyncs']}" if si["resyncs"] else ""))

    log(f"[2/6] Читаю {args.gnss}")
    fd = date.fromisoformat(args.date) if args.date else None
    g, gi = read_gnss(args.gnss, args.gnss_format, fd)
    log(f"  {gi['format']}: {gi['epochs']} эпох, {gi['rate_hz']} Гц, "
        f"RTK FIX {gi['fix_share'] * 100:.0f}%")
    crs, fwd, inv = make_proj(float(np.median(g["lon"])), float(np.median(g["lat"])))
    g["E"], g["N"] = fwd.transform(g["lon"], g["lat"])
    s["E_low"], s["N_low"] = fwd.transform(s["lon"], s["lat"])
    mot = gnss_motion(g, args.grid_dt, args.max_gap, min_speed=0.3)

    log("[3/6] Синхронизация времени")
    if args.offset is not None:
        tm = dict(a=args.offset, b=args.drift_ppm * 1e-6,
                  t_mid=0.5 * (s["t_rel"][0] + s["t_rel"][-1]), manual=True)
        log(f"  смещение задано вручную: {args.offset} с, дрейф {args.drift_ppm} ppm")
    else:
        tm = estimate_time_model(s, g, mot, args, rep)
        if tm["edge"]:
            log("  ! минимум на краю диапазона поиска — результат ненадёжен")
    t_abs = s["t_rel"] + tm["a"] + tm["b"] * (s["t_rel"] - tm["t_mid"])
    res.update(offset_s=round(tm["a"], 3), drift_ppm=round(tm["b"] * 1e6, 2),
               offset_ref_t_rel=round(tm["t_mid"], 3))
    overlap = np.mean((t_abs >= g["t"][0]) & (t_abs <= g["t"][-1]))
    log(f"  начало записи sl2: "
        f"{datetime.fromtimestamp(t_abs[0], timezone.utc):%Y-%m-%d %H:%M:%S} UTC, "
        f"покрыто GNSS: {overlap * 100:.0f}%")

    log("[4/6] Глубины и фильтрация")
    d = s["depth_m"].copy()
    ok = np.isfinite(d) & (d > args.min_depth) & (d < args.max_depth)
    n0 = int(ok.sum())
    idx = np.where(ok)[0]
    spk = despike(d[idx], args.despike_window, args.despike_abs, args.despike_rel)
    ok[idx[~spk]] = False
    log(f"  валидных глубин {n0} из {len(d)}, отброшено выбросов {int((~spk).sum())}")
    d_tot = d + args.draft

    log("[5/6] Задержка глубины")
    L = args.latency
    if args.auto_latency:
        L, scan, contrast, edge = scan_latency(t_abs[ok], d_tot[ok], g, mot, args)
        rep["latency_scan"] = scan
        log(f"  подобрана задержка: {L:+.2f} с (контраст минимума {contrast * 100:.0f}%)")
        if edge or contrast < 0.05:
            log("  ! минимум слабо выражен — нужны встречные или пересекающиеся галсы "
                "над склонами. Лучше задать --latency по тестовому выходу.")
    else:
        log(f"  задержка задана: {L:+.2f} с")
    res["latency_s"] = round(L, 3)

    E, N, h, q, hdg = ping_positions(t_abs - L, g, mot, args.lever, args.max_gap)
    minq = {"any": {1, 2, 4, 5, 6}, "float": {4, 5}, "fix": {4}}[args.min_fix]
    ok &= np.isfinite(E) & np.isin(q, list(minq))
    z = None
    if args.antenna_height is not None:
        z = h - args.antenna_height - d_tot
    lon, lat = inv.transform(E, N)

    log(f"[6/6] Запись результатов в {out}")
    sel = np.where(ok)[0]
    cols = {
        "time_utc": [datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds")
                     .replace("+00:00", "Z") for t in t_abs[sel]],
        "lat": np.round(lat[sel], 8), "lon": np.round(lon[sel], 8),
        f"E_{crs.to_epsg()}": np.round(E[sel], 3), f"N_{crs.to_epsg()}": np.round(N[sel], 3),
        "depth_m": np.round(d_tot[sel], 3),
    }
    if z is not None:
        cols["z_bottom"] = np.round(z[sel], 3)
    cols.update(gnss_q=q[sel], heading=np.round(hdg[sel], 1),
                sl2_t_rel=np.round(s["t_rel"][sel], 3))
    write_points(base + "_points.csv", cols)
    files = [base + "_points.csv"]

    V, vname = (z[sel], "отметка дна, м") if z is not None else (d_tot[sel], "глубина, м")
    if not args.no_grid and len(sel) > 10:
        gf, _ = make_grid(E[sel], N[sel], V, args.grid_cell, args.max_interp, crs,
                          base + ("_zbottom" if z is not None else "_depth"))
        files += gf

    res.update(sl2=si, gnss=gi, crs=f"EPSG:{crs.to_epsg()}", points=int(len(sel)),
               lever_fwd_right_m=args.lever, draft_m=args.draft,
               antenna_height_m=args.antenna_height,
               depth_range_m=[round(float(d_tot[sel].min()), 2),
                              round(float(d_tot[sel].max()), 2)] if len(sel) else None)
    if "rms_pos" in tm:
        res.update(coarse_offset_s=round(tm["coarse"], 3), speed_ncc=round(tm["ncc"], 3),
                   lowrance_vs_gnss_rms_m=round(tm["rms_pos"], 2))
    with open(base + "_summary.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    files.append(base + "_summary.json")

    if "offset_scan" in rep:
        fixm = s["has_gps"]
        rep.update(mot_t=mot["t"], mot_speed=mot["speed"],
                   son_t=t_abs[fixm], son_speed=s["speed_ms"][fixm])
        rep.update(E=E[sel], N=N[sel], V=V, vname=vname)
        if make_report(base + "_report.png", rep, res):
            files.append(base + "_report.png")

    for f in files:
        log(f"  → {f}")
    log(f"\nЗадержку {res['latency_s']:+.2f} с можно задавать для других записей с той же "
        f"связкой и настройками эхолота: --latency {res['latency_s']}")


if __name__ == "__main__":
    main()
