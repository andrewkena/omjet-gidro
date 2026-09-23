#!/usr/bin/env python3
"""
Синтетический тест для sl2sync: галсы над склоном, известные смещение, дрейф и задержки.
Создаёт test.sl2, test_gnss.nmea и truth.json.
"""
import json
import math
import struct
from datetime import datetime, timezone

import numpy as np
from pyproj import Transformer

rng = np.random.default_rng(1)
R = 6356752.3142
LAT0, LON0 = 48.20, 16.37
fwd = Transformer.from_crs(4326, 32633, always_xy=True)
inv = Transformer.from_crs(32633, 4326, always_xy=True)
E0, N0 = fwd.transform(LON0, LAT0)

TRUE = dict(tau=1_789_900_000.0 - 5.0,   # абсолютное время старта минус t_rel старта
            drift_ppm=25.0, sonar_lag=0.8, lowrance_gps_lag=0.5,
            lever=[0.5, 0.3], draft=0.1, ant_h=1.0, water=168.0)


def depth(E, N):   # истинная глубина от поверхности
    x, y = E - E0, N - N0
    return 3 + 4 / (1 + np.exp(-(x - 100) / 6)) + 1.2 * np.sin(y / 25) + 0.5 * np.cos(x / 30)


# --- траектория: 12 галсов туда-обратно по E с разворотами ---
pts = []
for k in range(12):
    y = k * 8.0
    xs = np.linspace(0, 200, 400) if k % 2 == 0 else np.linspace(200, 0, 400)
    pts += [(x, y) for x in xs]
    if k < 11:
        cx = 200 if k % 2 == 0 else 0
        for a in np.linspace(-np.pi / 2, np.pi / 2, 40)[1:-1]:
            pts.append((cx + (4 * np.cos(a) if k % 2 == 0 else -4 * np.cos(a)), y + 4 + 4 * np.sin(a)))
# 4 поперечных галса для пересечений
for x in (40, 90, 110, 160):
    pts += [(x, y) for y in np.linspace(90, -2, 200)]
    pts += [(x + 1, y) for y in np.linspace(-2, 90, 3)]
P = np.array(pts)
s_arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])

dt = 0.02
T0 = 1_789_900_000.0
t, s_now, ts, ss = 0.0, 0.0, [], []
while s_now < s_arc[-1]:
    v = 1.8 + 0.5 * math.sin(t / 23) + 0.3 * math.sin(t / 7.3)
    ts.append(t); ss.append(s_now)
    s_now += v * dt; t += dt
ts, ss = np.array(ts) + T0, np.array(ss)
AE = E0 + np.interp(ss, s_arc, P[:, 0]); AN = N0 + np.interp(ss, s_arc, P[:, 1])
vE, vN = np.gradient(AE, dt), np.gradient(AN, dt)
hd = np.arctan2(vE, vN)


def at(tq, arr):
    return np.interp(tq, ts, arr)


def body(tq, fwd_, right_):
    h = np.arctan2(at(tq, np.sin(hd)), at(tq, np.cos(hd)))
    return (at(tq, AE) + fwd_ * np.sin(h) + right_ * np.cos(h),
            at(tq, AN) + fwd_ * np.cos(h) - right_ * np.sin(h))


# --- GNSS NMEA 5 Гц, старт на 20 с раньше эхолота ---
def dm(v, pos, neg, w):
    hemi = pos if v >= 0 else neg
    v = abs(v); d = int(v); m = (v - d) * 60
    return f"{d:0{w}d}{m:08.5f}", hemi


def cs(b):
    c = 0
    for ch in b:
        c ^= ord(ch)
    return f"{c:02X}"


lines = []
for tg in np.arange(ts[0] + 0.0, ts[-1], 0.2):
    lon, lat = inv.transform(at(tg, AE) + rng.normal(0, .01), at(tg, AN) + rng.normal(0, .01))
    dt_ = datetime.fromtimestamp(tg, timezone.utc)
    hms = dt_.strftime("%H%M%S") + f".{int(round(dt_.microsecond / 1e4)) % 100:02d}"
    la, lah = dm(lat, "N", "S", 2); lo, loh = dm(lon, "E", "W", 3)
    alt = TRUE["water"] + TRUE["ant_h"] + rng.normal(0, .01)
    b = f"GNGGA,{hms},{la},{lah},{lo},{loh},4,20,0.6,{alt:.3f},M,45.0,M,1.0,0000"
    lines.append(f"$" + b + "*" + cs(b))
    if abs(tg * 5 - round(tg * 5)) < 1e-6 and round(tg * 5) % 5 == 0:
        b = f"GNRMC,{hms},A,{la},{lah},{lo},{loh},3.5,90.0,{dt_:%d%m%y},,,D"
        lines.append(f"$" + b + "*" + cs(b))
open("test_gnss.nmea", "w").write("\r\n".join(lines) + "\r\n")

# --- SL2: пинги 15 Гц, канал 0 + часть кадров канала 2 ---
T_s0, T_s1 = ts[0] + 20, ts[-1] - 20
pings = np.arange(T_s0, T_s1, 1 / 15)
buf = bytearray(struct.pack("<HHHH", 2, 0, 3200, 0))
PK = 64
idx = 0
last_fix_T = None
for i, T in enumerate(pings):
    t_rel = (T - T_s0) * (1 + TRUE["drift_ppm"] * 1e-6) + 5.0
    fixT = math.floor(T) - TRUE["lowrance_gps_lag"]          # GPS Lowrance 1 Гц с запаздыванием
    lE, lN = body(fixT, -0.2, 0.0)
    lE += 1.0 + rng.normal(0, .3); lN += -0.5 + rng.normal(0, .3)
    if fixT == last_fix_T:
        lE, lN = prev_l
    prev_l, last_fix_T = (lE, lN), fixT
    lon, lat = inv.transform(lE, lN)
    spd = math.hypot(at(fixT, vE), at(fixT, vN))
    xE, xN = body(T - TRUE["sonar_lag"], *TRUE["lever"])
    dep = depth(xE, xN) - TRUE["draft"] + rng.normal(0, 0.03)
    if rng.random() < 0.005:
        dep = rng.choice([0.25, 60.0])
    for ch in ((0, 2) if i % 3 == 0 else (0,)):
        fr = bytearray(144 + PK)
        pos = len(buf)
        struct.pack_into("<I", fr, 0, pos)
        struct.pack_into("<H", fr, 28, len(fr))
        struct.pack_into("<H", fr, 32, ch)
        struct.pack_into("<H", fr, 34, PK)
        struct.pack_into("<I", fr, 36, idx)
        struct.pack_into("<f", fr, 64, dep / 0.3048)
        struct.pack_into("<f", fr, 100, spd / 0.514444)
        struct.pack_into("<i", fr, 108, int(round(math.radians(lon) * R)))
        struct.pack_into("<i", fr, 112, int(round(R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))))
        struct.pack_into("<I", fr, 140, int(round(t_rel * 1000)))
        buf += fr
        idx += 1
open("test.sl2", "wb").write(buf)

TRUE["tau_rel0"] = T_s0 - 5.0
json.dump(TRUE, open("truth.json", "w"), indent=2)
print(f"пингов {len(pings)}, длительность {(T_s1 - T_s0) / 60:.1f} мин, GNSS эпох {len(lines)}")
