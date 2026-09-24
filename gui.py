#!/usr/bin/env python3
"""
Главное окно sl2sync: выбор файла эхограммы, карта с точками (встроенный GPS
Lowrance) поверх выбранной онлайн-подложки, расчёт временного смещения
относительно GNSS-трека со сравнением треков до/после синхронизации.
"""
import itertools
import json
import os
import sys
import tempfile
import threading
import urllib.request
from datetime import datetime, timedelta

import numpy as np
from PySide6.QtCore import QRect, QSettings, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPainter, QPixmap
from PySide6.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                                QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                                QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                                QMainWindow, QMessageBox, QPushButton, QRadioButton,
                                QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget)

from sl2sync import (build_parser, estimate_time_model, gnss_motion, make_proj,
                      ping_positions, read_gnss, read_sl2)

APP_VERSION = "0.1.0"
GITHUB_REPO = "andrewkena/omjet-gidro"

BASEMAPS = {
    "Google Спутник": dict(
        url="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        max_zoom=20),
    "Google Карта": dict(
        url="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        max_zoom=20),
}


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def new_map_view():
    view = QWebEngineView()
    view.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
    return view


_html_seq = itertools.count()


def load_html(view, html, name):
    """QWebEnginePage.setHtml() молча обрезает контент больше ~2 МБ — для больших
    треков (десятки тысяч точек) грузим через временный файл, там лимита нет.
    Имя файла каждый раз новое: Chromium кэширует file:// по пути и не видит
    изменений в перезаписанном файле с тем же именем."""
    path = os.path.join(tempfile.gettempdir(), f"sl2sync_{name}_{next(_html_seq)}.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    view.load(QUrl.fromLocalFile(path))


def build_map_html(basemap, points, color_by_depth=True, vmin=None, vmax=None, steps=32,
                    axis_mode="time"):
    tile = BASEMAPS[basemap]
    lat, lon, val = points["lat"], points["lon"], points["value"]
    axis_key = "dist" if axis_mode == "distance" else "t_rel"
    axis_vals = points.get(axis_key) or list(range(len(val)))
    if lat:
        center = [sum(lat) / len(lat), sum(lon) / len(lon)]
    else:
        center = [0, 0]
    if vmin is None or vmax is None:
        vmin, vmax = (min(val), max(val)) if val else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"v": v, "i": i},
             "geometry": {"type": "Point", "coordinates": [lo, la]}}
            for i, (la, lo, v) in enumerate(zip(lat, lon, val))
        ],
    }
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body{{height:100%;margin:0;display:flex;flex-direction:column}}
#map{{flex:1;min-height:0}}
#timeline{{height:70px;width:100%;display:block;background:#151515;cursor:crosshair}}
.depth-legend{{background:rgba(20,20,20,.65);color:#fff;padding:8px;border-radius:4px;
               font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.4);
               display:flex;flex-direction:column;align-items:center}}
.depth-legend .bar{{width:12px;height:120px;
               background:linear-gradient(to top, rgb(128,0,0), rgb(255,0,0), rgb(255,255,0),
               rgb(0,255,255), rgb(0,0,255), rgb(0,0,143))}}
.depth-legend .scale{{display:flex;flex-direction:column;justify-content:space-between;
               height:120px;text-align:center}}
.depth-legend .row{{display:flex;align-items:stretch}}
.depth-cursor-tip{{background:#222;color:#fff;border:none;font:12px sans-serif}}</style>
</head><body>
<div id="map"></div>
<canvas id="timeline"></canvas>
<script>
var map = L.map('map', {{preferCanvas: true, attributionControl: false}}).setView([{center[0]}, {center[1]}], 15);
L.tileLayer('{tile["url"]}', {{
  maxZoom: {tile["max_zoom"]}
}}).addTo(map);
var data = {json.dumps(geojson)};
var pts = data.features.map(function (f) {{
  return {{lat: f.geometry.coordinates[1], lon: f.geometry.coordinates[0], v: f.properties.v}};
}});
var axisVals = {json.dumps(list(axis_vals))};
var axisMode = {json.dumps(axis_mode)};
var vmin = {vmin}, vmax = {vmax}, steps = {max(2, int(steps))};
var colorByDepth = {"true" if color_by_depth else "false"};
var jetStops = [
  [0.0, [0, 0, 143]], [0.125, [0, 0, 255]], [0.375, [0, 255, 255]],
  [0.625, [255, 255, 0]], [0.875, [255, 0, 0]], [1.0, [128, 0, 0]]
];
function jetColor(t) {{
  for (var i = 0; i < jetStops.length - 1; i++) {{
    var a = jetStops[i], b = jetStops[i + 1];
    if (t <= b[0] || i === jetStops.length - 2) {{
      var f = (t - a[0]) / (b[0] - a[0]);
      return 'rgb(' + Math.round(a[1][0] + f * (b[1][0] - a[1][0])) + ',' +
                       Math.round(a[1][1] + f * (b[1][1] - a[1][1])) + ',' +
                       Math.round(a[1][2] + f * (b[1][2] - a[1][2])) + ')';
    }}
  }}
}}
function color(v) {{
  if (!colorByDepth) {{ return '#3388ff'; }}
  var t = Math.max(0, Math.min(1, (v - vmin) / (vmax - vmin)));
  var band = Math.min(steps - 1, Math.floor(t * steps));
  t = band / (steps - 1);
  return jetColor(1 - t);
}}
var layer = L.geoJSON(data, {{
  pointToLayer: function (f, latlng) {{
    var m = L.circleMarker(latlng, {{radius: 3, weight: 0, fillOpacity: 0.85,
                                      fillColor: color(f.properties.v), color: color(f.properties.v)}});
    m.bindTooltip(f.properties.v.toFixed(2) + ' м', {{sticky: true}});
    m.on('mouseover', function () {{ drawTimeline(f.properties.i); }});
    return m;
  }}
}}).addTo(map);

var cursorMarker = L.circleMarker([0, 0], {{radius: 7, color: '#fff', weight: 2,
                                             fillColor: '#ffd400', fillOpacity: 1}});
cursorMarker.bindTooltip('', {{permanent: true, direction: 'top', className: 'depth-cursor-tip'}});

var timelineCanvas = document.getElementById('timeline');
var tctx = timelineCanvas.getContext('2d');

function fitTimelineCanvas() {{
  timelineCanvas.width = timelineCanvas.clientWidth;
  timelineCanvas.height = timelineCanvas.clientHeight;
}}

function formatAxisLabel(v) {{
  if (axisMode === 'distance') {{
    return v >= 1000 ? (v / 1000).toFixed(1) + ' км' : Math.round(v) + ' м';
  }}
  var s = Math.round(v);
  var m = Math.floor(s / 60), sec = s % 60;
  return m + ':' + (sec < 10 ? '0' : '') + sec;
}}

function drawTimeline(cursorIndex) {{
  var w = timelineCanvas.width, h = timelineCanvas.height;
  var axisH = 14, plotH = h - axisH;
  tctx.clearRect(0, 0, w, h);
  if (!pts.length) {{ return; }}
  var a0 = axisVals[0], a1 = axisVals[axisVals.length - 1];
  var nTicks = Math.max(2, Math.min(8, Math.round(w / 90)));
  tctx.strokeStyle = 'rgba(255,255,255,.15)';
  tctx.fillStyle = '#aaa';
  tctx.font = '10px sans-serif';
  tctx.lineWidth = 1;
  for (var k = 0; k <= nTicks; k++) {{
    var frac = k / nTicks;
    var x = frac * w;
    tctx.beginPath();
    tctx.moveTo(x, 0);
    tctx.lineTo(x, plotH);
    tctx.stroke();
    var label = formatAxisLabel(a0 + frac * (a1 - a0));
    tctx.textAlign = k === 0 ? 'left' : (k === nTicks ? 'right' : 'center');
    tctx.fillText(label, Math.min(w - 2, Math.max(2, x)), h - 3);
  }}
  tctx.strokeStyle = '#3388ff';
  tctx.lineWidth = 1;
  tctx.beginPath();
  for (var i = 0; i < pts.length; i++) {{
    var x = pts.length > 1 ? i / (pts.length - 1) * w : 0;
    var t = Math.max(0, Math.min(1, (pts[i].v - vmin) / (vmax - vmin)));
    var y = 4 + t * (plotH - 8);
    if (i === 0) {{ tctx.moveTo(x, y); }} else {{ tctx.lineTo(x, y); }}
  }}
  tctx.stroke();
  if (cursorIndex != null) {{
    var cx = pts.length > 1 ? cursorIndex / (pts.length - 1) * w : 0;
    tctx.strokeStyle = '#ffd400';
    tctx.lineWidth = 2;
    tctx.beginPath();
    tctx.moveTo(cx, 0);
    tctx.lineTo(cx, plotH);
    tctx.stroke();
  }}
}}

function showAtIndex(i) {{
  var p = pts[Math.max(0, Math.min(pts.length - 1, i))];
  if (!cursorMarker._map) {{ cursorMarker.addTo(map); }}
  cursorMarker.setLatLng([p.lat, p.lon]);
  cursorMarker.setTooltipContent(p.v.toFixed(2) + ' м');
  cursorMarker.openTooltip();
  drawTimeline(i);
}}

function indexFromEvent(e) {{
  var rect = timelineCanvas.getBoundingClientRect();
  var x = e.clientX - rect.left;
  return Math.round(x / rect.width * (pts.length - 1));
}}

timelineCanvas.addEventListener('mousemove', function (e) {{
  if (pts.length) {{ showAtIndex(indexFromEvent(e)); }}
}});
timelineCanvas.addEventListener('mouseleave', function () {{ drawTimeline(null); }});

fitTimelineCanvas();
drawTimeline(null);
window.addEventListener('resize', function () {{ fitTimelineCanvas(); drawTimeline(null); }});

if (data.features.length) {{
  map.fitBounds(layer.getBounds(), {{maxZoom: 18}});
  if (colorByDepth) {{
    var legend = L.control({{position: 'bottomright'}});
    legend.onAdd = function () {{
      var div = L.DomUtil.create('div', 'depth-legend');
      div.innerHTML = '<div>Глубина, м</div><div class="row">' +
        '<div class="bar"></div>' +
        '<div class="scale"><span>' + vmax.toFixed(1) + '</span><span>' + vmin.toFixed(1) + '</span></div>' +
        '</div>';
      return div;
    }};
    legend.addTo(map);
  }}
}}
</script>
</body></html>"""


def build_tracks_html(basemap, tracks):
    tile = BASEMAPS[basemap]
    all_lat = [la for t in tracks for la in t["lat"]]
    all_lon = [lo for t in tracks for lo in t["lon"]]
    center = [sum(all_lat) / len(all_lat), sum(all_lon) / len(all_lon)] if all_lat else [0, 0]
    lines = []
    for t in tracks:
        coords = list(zip(t["lat"], t["lon"]))
        lines.append(
            f"L.polyline({json.dumps(coords)}, {{color: '{t['color']}', weight: 2}})"
            f".bindTooltip({json.dumps(t['name'])}).addTo(map);")
    bounds = list(zip(all_lat, all_lon))
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}}</style>
</head><body>
<div id="map"></div>
<script>
var map = L.map('map', {{preferCanvas: true, attributionControl: false}}).setView([{center[0]}, {center[1]}], 15);
L.tileLayer('{tile["url"]}', {{maxZoom: {tile["max_zoom"]}}}).addTo(map);
{chr(10).join(lines)}
var bounds = {json.dumps(bounds)};
if (bounds.length) {{ map.fitBounds(bounds, {{maxZoom: 18}}); }}
</script>
</body></html>"""


def run_async(target_fn, on_finished, on_error, poll_ms=50):
    """Выполняет target_fn() в отдельном потоке и опрашивает его через QTimer.
    В этом окружении QThread + сигнал между потоками, чей обработчик обращается
    к QWebEngineView, приводит к падению процесса (воспроизведено и проверено) —
    поэтому вместо QThread/Signal используется обычный threading.Thread и
    периодический опрос из основного потока, без сигналов между потоками."""
    result = {}

    def worker():
        try:
            result["value"] = target_fn()
        except SystemExit as e:
            result["error"] = str(e)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"

    thread = threading.Thread(target=worker, daemon=True)

    def poll():
        if thread.is_alive():
            QTimer.singleShot(poll_ms, poll)
            return
        if "error" in result:
            on_error(result["error"])
        else:
            on_finished(result["value"])

    thread.start()
    QTimer.singleShot(poll_ms, poll)


def cumulative_distance(lat, lon):
    """Приближённое расстояние вдоль трека, м (эквидистантная проекция — для
    масштаба одного выхода лодки точнее не требуется)."""
    if len(lat) < 2:
        return [0.0] * len(lat)
    R = 6371000.0
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    dlat = np.diff(lat_r)
    dlon = np.diff(lon_r)
    mean_lat = (lat_r[:-1] + lat_r[1:]) / 2
    seg = np.hypot(dlon * np.cos(mean_lat) * R, dlat * R)
    return np.concatenate([[0.0], np.cumsum(seg)]).tolist()


def read_echogram_file(sl2_path):
    s, info = read_sl2(sl2_path)
    m = s["has_gps"] & np.isfinite(s["depth_m"]) & (s["depth_m"] > 0)
    lat, lon = s["lat"][m], s["lon"][m]
    t_rel = s["t_rel"][m]
    duration_s = info["duration_s"]
    # sl2 хранит только относительное время (см. docs/CONTEXT.md) — абсолютное
    # время берём от mtime файла (момент завершения записи) и отсчитываем назад.
    end_dt = datetime.fromtimestamp(os.path.getmtime(sl2_path))
    start_dt = end_dt - timedelta(seconds=duration_s)
    return dict(path=sl2_path, lat=lat.tolist(), lon=lon.tolist(),
                depth=s["depth_m"][m].tolist(), t_rel=t_rel.tolist(),
                dist=cumulative_distance(lat, lon),
                start=start_dt.isoformat(), end=end_dt.isoformat(), duration_s=duration_s)


def read_echogram_waterfall(sl2_path):
    """Сырые сэмплы эхограммы (амплитуда сонара) по каждому пингу — байты сразу
    после 144-байтного заголовка кадра. Формат байтов документально не описан
    в docs/CONTEXT.md и не проверен на реальном железе (см. открытые вопросы
    там же) — по общепринятому для sl2 предположению это 8-битная амплитуда
    по глубине, старт столбца сверху (поверхность) вниз (дно)."""
    s, info = read_sl2(sl2_path, with_echogram=True)
    # В пределах одного канала могут чередоваться пинги разных частот (например,
    # CHIRP low/high) — вперемешку они дают полосатую/рваную картинку, поэтому
    # оставляем только самую частую частоту в этом канале.
    freqs, counts = np.unique(s["freq"], return_counts=True)
    dominant = freqs[np.argmax(counts)]
    mixed_freqs = len(freqs)
    m = s["freq"] == dominant
    records = [s["echogram"][i] for i in np.where(m)[0]]
    return dict(records=records, depth_m=s["depth_m"][m].tolist(),
                t_rel=s["t_rel"][m].tolist(),
                dist=cumulative_distance(s["lat"][m], s["lon"][m]),
                mixed_freqs=mixed_freqs)


def format_axis_label(v, axis_mode):
    if axis_mode == "distance":
        return f"{v / 1000:.1f} км" if v >= 1000 else f"{v:.0f} м"
    m, sec = divmod(int(round(v)), 60)
    return f"{m}:{sec:02d}"


def build_echogram_image(records, depth_m=None):
    """Каждый пинг может писаться с своим диапазоном (автодиапазон сонара) —
    номер байта сам по себе НЕ соответствует одной и той же глубине в разных
    пингах. Если известна глубина дна по пингу (depth_m, из заголовка кадра),
    растягиваем/сжимаем сырые байты каждого столбца так, будто они снятые на
    диапазон 0..depth_m[i], и ресэмплим на общую сетку 0..max(depth_m) —
    это даёт единый вертикальный масштаб по всей картинке. Без этого разные
    диапазоны дают рваную, скачущую по глубине картинку."""
    lengths = [len(r) for r in records if r]
    if not lengths:
        return None
    rows = max(lengths)
    max_depth = max(depth_m) if depth_m else 0.0
    if not depth_m or max_depth <= 0:
        arr = np.zeros((rows, len(records)), dtype=np.uint8)
        for col, r in enumerate(records):
            if r:
                arr[:len(r), col] = np.frombuffer(r, dtype=np.uint8)
        return arr

    target_y = np.linspace(0.0, max_depth, rows)
    arr = np.zeros((rows, len(records)), dtype=np.uint8)
    for col, (r, d) in enumerate(zip(records, depth_m)):
        if r and d > 0:
            src = np.frombuffer(r, dtype=np.uint8).astype(np.float32)
            src_depth = np.linspace(0.0, d, len(src))
            arr[:, col] = np.interp(target_y, src_depth, src, left=0.0, right=0.0)
    return arr


def array_to_qimage(arr):
    h, w = arr.shape
    arr = np.ascontiguousarray(arr)
    img = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
    return img.copy()


def _version_tuple(v):
    out = []
    for p in v.split("."):
        try:
            out.append(int(p))
        except ValueError:
            out.append(0)
    return tuple(out)


def fetch_latest_release():
    req = urllib.request.Request(
        f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
        headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=6) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    tag = str(data.get("tag_name", "")).lstrip("vV")
    has_update = bool(tag) and _version_tuple(tag) > _version_tuple(APP_VERSION)
    return dict(has_update=has_update, tag=data.get("tag_name", ""),
                url=data.get("html_url", ""), notes=data.get("body") or "")


def merge_gnss_tracks(gnss_paths):
    tracks = [read_gnss(p)[0] for p in gnss_paths]
    if len(tracks) == 1:
        return tracks[0]
    merged = {k: np.concatenate([t[k] for t in tracks]) for k in tracks[0]}
    order = np.argsort(merged["t"], kind="stable")
    return {k: v[order] for k, v in merged.items()}


def compute_time_offset(sl2_path, gnss_paths):
    if isinstance(gnss_paths, str):
        gnss_paths = [gnss_paths]
    args = build_parser().parse_args([sl2_path, gnss_paths[0]])
    s, si = read_sl2(sl2_path)
    g = merge_gnss_tracks(gnss_paths)
    crs, fwd, inv = make_proj(float(np.median(g["lon"])), float(np.median(g["lat"])))
    g["E"], g["N"] = fwd.transform(g["lon"], g["lat"])
    s["E_low"], s["N_low"] = fwd.transform(s["lon"], s["lat"])
    mot = gnss_motion(g, args.grid_dt, args.max_gap, min_speed=0.3)
    tm = estimate_time_model(s, g, mot, args, {})
    t_abs = s["t_rel"] + tm["a"] + tm["b"] * (s["t_rel"] - tm["t_mid"])
    E2, N2, h, q, hdg = ping_positions(t_abs, g, mot, [0.0, 0.0], args.max_gap)
    ok = np.isfinite(E2)
    lon2, lat2 = inv.transform(E2[ok], N2[ok])
    has_gps = s["has_gps"]
    return dict(
        offset_s=tm["a"], rms_pos=tm.get("rms_pos"),
        before=dict(
            sl2=dict(lat=s["lat"][has_gps].tolist(), lon=s["lon"][has_gps].tolist()),
            gnss=dict(lat=g["lat"].tolist(), lon=g["lon"].tolist())),
        after=dict(
            sl2=dict(lat=lat2.tolist(), lon=lon2.tolist()),
            gnss=dict(lat=g["lat"].tolist(), lon=g["lon"].tolist())),
    )


class EchogramCanvas(QWidget):
    RULER_W = 55
    AXIS_H = 22

    def __init__(self):
        super().__init__()
        self.arr = None
        self.image = None
        self.depth_m = []
        self.axis_vals = []
        self.axis_mode = "time"
        self.zoom = 1.0
        self.contrast = 1.0
        self.setMouseTracking(True)

    def set_data(self, arr, depth_m, axis_vals, axis_mode):
        self.arr = arr
        self.depth_m = depth_m
        self.axis_vals = axis_vals or list(range(arr.shape[1]))
        self.axis_mode = axis_mode
        self._rebuild_image()
        self.updateGeometry()
        self.resize(self.sizeHint())
        self.update()

    def set_contrast(self, value):
        self.contrast = value
        self._rebuild_image()
        self.update()

    def _rebuild_image(self):
        if self.arr is None:
            return
        adj = np.clip((self.arr.astype(np.float32) - 128.0) * self.contrast + 128.0, 0, 255)
        self.image = array_to_qimage(adj.astype(np.uint8))

    def sizeHint(self):
        if self.image is None:
            return QSize(400, 200)
        return QSize(self.RULER_W + int(self.image.width() * self.zoom),
                     int(self.image.height() * self.zoom) + self.AXIS_H)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("black"))
        if self.image is None:
            painter.setPen(QColor("white"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Чтение файла…")
            return
        w = int(self.image.width() * self.zoom)
        h = int(self.image.height() * self.zoom)
        painter.drawImage(QRect(self.RULER_W, 0, w, h), self.image)
        self._draw_depth_ruler(painter, h)
        self._draw_axis(painter, w, h)

    def _draw_depth_ruler(self, painter, h):
        painter.setPen(QColor("#ccc"))
        max_depth = max(self.depth_m) if self.depth_m else 0.0
        ticks = max(2, min(10, h // 40))
        for k in range(ticks + 1):
            frac = k / ticks
            y = frac * h
            painter.drawLine(self.RULER_W - 5, int(y), self.RULER_W, int(y))
            label = f"{frac * max_depth:.1f}"
            painter.drawText(2, min(int(y) + 4, h), label)

    def _draw_axis(self, painter, w, h):
        painter.setPen(QColor("#ccc"))
        if len(self.axis_vals) < 2:
            return
        a0, a1 = self.axis_vals[0], self.axis_vals[-1]
        ticks = max(2, min(10, w // 90))
        for k in range(ticks + 1):
            frac = k / ticks
            x = self.RULER_W + frac * w
            label = format_axis_label(a0 + frac * (a1 - a0), self.axis_mode)
            painter.drawLine(int(x), h, int(x), h + 4)
            painter.drawText(int(x) - 15, h + self.AXIS_H - 4, label)

    def wheelEvent(self, event):
        if self.image is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = max(0.1, min(20.0, self.zoom * factor))
        self.updateGeometry()
        self.resize(self.sizeHint())
        self.update()
        event.accept()


class EchogramViewDialog(QDialog):
    def __init__(self, parent, sl2_path, axis_mode="time"):
        super().__init__(parent)
        self.setWindowTitle(f"Эхограмма — {os.path.basename(sl2_path)}")
        self.resize(1000, 500)
        self.axis_mode = axis_mode

        self.status_label = QLabel("Чтение файла…")
        self.canvas = EchogramCanvas()
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)

        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(20, 400)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(
            lambda v: self.canvas.set_contrast(v / 100.0))
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(QLabel("Контраст:"))
        contrast_row.addWidget(self.contrast_slider)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(scroll, 1)
        layout.addLayout(contrast_row)
        self.setLayout(layout)

        run_async(lambda: read_echogram_waterfall(sl2_path),
                  on_finished=self.on_loaded, on_error=self.on_error)

    def on_loaded(self, data):
        arr = build_echogram_image(data["records"], data["depth_m"])
        if arr is None:
            self.status_label.setText("В файле нет данных эхограммы")
            return
        note = ""
        if data["mixed_freqs"] > 1:
            note = (f" — в канале {data['mixed_freqs']} частоты, показана самая частая, "
                    f"остальные пинги пропущены")
        self.status_label.setText(f"{arr.shape[1]} пингов, {arr.shape[0]} байт по глубине "
                                   f"(колесо мыши — масштаб){note}")
        axis_vals = data["dist"] if self.axis_mode == "distance" else data["t_rel"]
        self.canvas.set_data(arr, data["depth_m"], axis_vals, self.axis_mode)

    def on_error(self, message):
        self.status_label.setText(f"Ошибка чтения: {message}")


class OffsetDialog(QDialog):
    def __init__(self, parent, basemap, result):
        super().__init__(parent)
        self.setWindowTitle("Сравнение треков до/после синхронизации")
        self.resize(1400, 800)

        info = f"Смещение времени: {result['offset_s']:+.3f} с"
        if result.get("rms_pos") is not None:
            info += f"    СКО позиций после синхронизации: {result['rms_pos']:.2f} м"

        left_view = new_map_view()
        load_html(left_view, build_tracks_html(basemap, [
            dict(name="Эхограмма (сырой GPS)", color="red", **result["before"]["sl2"]),
            dict(name="GNSS-трек", color="blue", **result["before"]["gnss"]),
        ]), "before")
        right_view = new_map_view()
        load_html(right_view, build_tracks_html(basemap, [
            dict(name="Эхограмма (синхр.)", color="red", **result["after"]["sl2"]),
            dict(name="GNSS-трек", color="blue", **result["after"]["gnss"]),
        ]), "after")

        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("До синхронизации"))
        left_col.addWidget(left_view)
        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("После синхронизации"))
        right_col.addWidget(right_view)
        maps_row = QHBoxLayout()
        maps_row.addLayout(left_col)
        maps_row.addLayout(right_col)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(info))
        layout.addLayout(maps_row)
        self.setLayout(layout)


class SettingsDialog(QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Настройки")

        depth_group = QGroupBox("Окраска глубин")
        self.auto_radio = QRadioButton("Автоматическая (мин/макс берутся с эхограммы)")
        self.manual_radio = QRadioButton("Ручная")
        manual = settings.value("depth_mode", "auto") == "manual"
        self.manual_radio.setChecked(manual)
        self.auto_radio.setChecked(not manual)

        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(-1000.0, 1000.0)
        self.min_spin.setSuffix(" м")
        self.min_spin.setValue(float(settings.value("depth_min", 0.0)))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(-1000.0, 1000.0)
        self.max_spin.setSuffix(" м")
        self.max_spin.setValue(float(settings.value("depth_max", 10.0)))
        manual_row = QHBoxLayout()
        manual_row.addWidget(QLabel("Мин:"))
        manual_row.addWidget(self.min_spin)
        manual_row.addWidget(QLabel("Макс:"))
        manual_row.addWidget(self.max_spin)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(2, 128)
        self.steps_spin.setValue(int(settings.value("depth_steps", 32)))
        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Число ступеней окраски:"))
        steps_row.addWidget(self.steps_spin)

        def update_manual_enabled():
            self.min_spin.setEnabled(self.manual_radio.isChecked())
            self.max_spin.setEnabled(self.manual_radio.isChecked())

        self.auto_radio.toggled.connect(update_manual_enabled)
        update_manual_enabled()

        depth_layout = QVBoxLayout()
        depth_layout.addWidget(self.auto_radio)
        depth_layout.addWidget(self.manual_radio)
        depth_layout.addLayout(manual_row)
        depth_layout.addLayout(steps_row)
        depth_group.setLayout(depth_layout)

        axis_group = QGroupBox("Временная шкала")
        self.axis_time_radio = QRadioButton("Время")
        self.axis_dist_radio = QRadioButton("Расстояние")
        is_dist = settings.value("timeline_axis", "time") == "distance"
        self.axis_dist_radio.setChecked(is_dist)
        self.axis_time_radio.setChecked(not is_dist)
        axis_layout = QVBoxLayout()
        axis_layout.addWidget(self.axis_time_radio)
        axis_layout.addWidget(self.axis_dist_radio)
        axis_group.setLayout(axis_layout)

        cache_group = QGroupBox("Кэш")
        profile = QWebEngineProfile.defaultProfile()
        self.cache_label = QLabel(f"Текущий размер кэша: {dir_size(profile.cachePath()) / 1e6:.1f} МБ")
        clear_btn = QPushButton("Очистить кэш")
        clear_btn.clicked.connect(self.clear_cache)
        self.cache_limit_spin = QSpinBox()
        self.cache_limit_spin.setRange(1, 1_000_000)
        self.cache_limit_spin.setSuffix(" МБ")
        self.cache_limit_spin.setValue(int(settings.value("cache_max_mb", 1024)))
        limit_row = QHBoxLayout()
        limit_row.addWidget(QLabel("Отведённый объём под кэш карт:"))
        limit_row.addWidget(self.cache_limit_spin)

        cache_layout = QVBoxLayout()
        cache_layout.addWidget(self.cache_label)
        cache_layout.addWidget(clear_btn)
        cache_layout.addLayout(limit_row)
        cache_group.setLayout(cache_layout)

        update_row = QHBoxLayout()
        update_btn = QPushButton("Проверить обновления")
        update_btn.clicked.connect(lambda: parent.check_for_updates(silent=False))
        update_row.addWidget(update_btn)
        update_row.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(depth_group)
        layout.addWidget(axis_group)
        layout.addWidget(cache_group)
        layout.addLayout(update_row)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def clear_cache(self):
        profile = QWebEngineProfile.defaultProfile()
        profile.clearHttpCache()
        self.cache_label.setText(f"Текущий размер кэша: {dir_size(profile.cachePath()) / 1e6:.1f} МБ")

    def accept(self):
        self.settings.setValue("depth_mode", "manual" if self.manual_radio.isChecked() else "auto")
        self.settings.setValue("depth_min", self.min_spin.value())
        self.settings.setValue("depth_max", self.max_spin.value())
        self.settings.setValue("depth_steps", self.steps_spin.value())
        self.settings.setValue("timeline_axis", "distance" if self.axis_dist_radio.isChecked() else "time")
        cache_mb = self.cache_limit_spin.value()
        self.settings.setValue("cache_max_mb", cache_mb)
        QWebEngineProfile.defaultProfile().setHttpCacheMaximumSize(cache_mb * 1024 * 1024)
        super().accept()


class DepthOffsetDialog(QDialog):
    def __init__(self, parent, file_data, current_offsets):
        super().__init__(parent)
        self.setWindowTitle("Смещение по эхограммам")
        self.spins = {}
        form = QFormLayout()
        for f in file_data:
            spin = QDoubleSpinBox()
            spin.setRange(-1000.0, 1000.0)
            spin.setSuffix(" см")
            spin.setValue(current_offsets.get(f["path"], 0.0))
            self.spins[f["path"]] = spin
            form.addRow(os.path.basename(f["path"]), spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def values(self):
        return {path: spin.value() for path, spin in self.spins.items()}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ОМДЖЕТ Гидро")
        self.resize(1100, 800)
        self.settings = QSettings("sl2sync", "gui")
        QWebEngineProfile.defaultProfile().setHttpCacheMaximumSize(
            int(self.settings.value("cache_max_mb", 1024)) * 1024 * 1024)
        self.points = None
        self.file_data = []
        self.crop_mode = "time"
        self.depth_offset_mode = "all"
        self.depth_offset_per_file = {}

        self.sl2_edit = QLineEdit(readOnly=True)
        sl2_btn = QPushButton("Обзор…")
        sl2_btn.clicked.connect(self.pick_sl2)
        view_echogram_btn = QPushButton("Посмотреть эхограмму")
        view_echogram_btn.clicked.connect(self.view_echogram)
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setFrameShadow(QFrame.Shadow.Sunken)
        multi_btn = QPushButton("Мультиэхограмма")
        multi_btn.clicked.connect(self.pick_multi_sl2)
        sep1b = QFrame()
        sep1b.setFrameShape(QFrame.Shape.VLine)
        sep1b.setFrameShadow(QFrame.Shadow.Sunken)
        close_echograms_btn = QPushButton("Закрыть эхограммы")
        close_echograms_btn.clicked.connect(self.close_echograms)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Файл эхограммы (.sl2):"))
        row1.addWidget(self.sl2_edit, 1)
        row1.addWidget(sl2_btn)
        row1.addWidget(view_echogram_btn)
        row1.addWidget(sep1)
        row1.addWidget(multi_btn)
        row1.addWidget(sep1b)
        row1.addWidget(close_echograms_btn)

        self.gnss_paths = []
        self.gnss_edit = QLineEdit(readOnly=True)
        gnss_btn = QPushButton("Обзор…")
        gnss_btn.clicked.connect(self.pick_gnss)
        sep2a = QFrame()
        sep2a.setFrameShape(QFrame.Shape.VLine)
        sep2a.setFrameShadow(QFrame.Shadow.Sunken)
        multi_gnss_btn = QPushButton("Мультитрек")
        multi_gnss_btn.clicked.connect(self.pick_multi_gnss)
        sep2b = QFrame()
        sep2b.setFrameShape(QFrame.Shape.VLine)
        sep2b.setFrameShadow(QFrame.Shadow.Sunken)
        close_gnss_btn = QPushButton("Закрыть")
        close_gnss_btn.clicked.connect(self.close_gnss)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("GNSS-трек:"))
        row2.addWidget(self.gnss_edit, 1)
        row2.addWidget(gnss_btn)
        row2.addWidget(sep2a)
        row2.addWidget(multi_gnss_btn)
        row2.addWidget(sep2b)
        row2.addWidget(close_gnss_btn)

        self.offset_btn = QPushButton("Рассчитать смещение")
        self.offset_btn.setEnabled(False)
        self.offset_btn.clicked.connect(self.calc_offset)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        settings_btn = QPushButton("Настройки")
        settings_btn.clicked.connect(self.open_settings)
        row_offset = QHBoxLayout()
        row_offset.addStretch(1)
        row_offset.addWidget(self.offset_btn)
        row_offset.addWidget(sep2)
        row_offset.addWidget(settings_btn)

        self.map_view = new_map_view()
        load_html(self.map_view, build_map_html("Google Спутник", dict(lat=[], lon=[], value=[])), "main")

        crop_group = QGroupBox("Обрезка")
        self.crop_time_radio = QRadioButton("Время")
        self.crop_dist_radio = QRadioButton("Расстояние")
        self.crop_time_radio.setChecked(True)
        self.crop_time_radio.toggled.connect(self.on_crop_mode_toggled)

        self.crop_start_time_spin = QDoubleSpinBox()
        self.crop_start_time_spin.setRange(0.0, 1_000_000.0)
        self.crop_start_time_spin.setSuffix(" с")
        self.crop_end_time_spin = QDoubleSpinBox()
        self.crop_end_time_spin.setRange(0.0, 1_000_000.0)
        self.crop_end_time_spin.setSuffix(" с")
        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("С начала:"))
        time_row.addWidget(self.crop_start_time_spin)
        time_row.addWidget(QLabel("С конца:"))
        time_row.addWidget(self.crop_end_time_spin)

        self.crop_start_dist_spin = QDoubleSpinBox()
        self.crop_start_dist_spin.setRange(0.0, 1_000_000.0)
        self.crop_start_dist_spin.setSuffix(" м")
        self.crop_end_dist_spin = QDoubleSpinBox()
        self.crop_end_dist_spin.setRange(0.0, 1_000_000.0)
        self.crop_end_dist_spin.setSuffix(" м")
        dist_row = QHBoxLayout()
        dist_row.addWidget(QLabel("С начала:"))
        dist_row.addWidget(self.crop_start_dist_spin)
        dist_row.addWidget(QLabel("С конца:"))
        dist_row.addWidget(self.crop_end_dist_spin)

        for spin in (self.crop_start_time_spin, self.crop_end_time_spin,
                     self.crop_start_dist_spin, self.crop_end_dist_spin):
            spin.valueChanged.connect(self.on_crop_value_changed)

        crop_layout = QVBoxLayout()
        crop_layout.addWidget(self.crop_time_radio)
        crop_layout.addLayout(time_row)
        crop_layout.addWidget(self.crop_dist_radio)
        crop_layout.addLayout(dist_row)
        crop_group.setLayout(crop_layout)
        self.update_crop_enabled()

        offset_group = QGroupBox("Смещение")
        self.depth_offset_all_spin = QDoubleSpinBox()
        self.depth_offset_all_spin.setRange(-1000.0, 1000.0)
        self.depth_offset_all_spin.setSuffix(" см")
        self.depth_offset_all_spin.valueChanged.connect(self.on_depth_offset_all_changed)
        offset_value_row = QHBoxLayout()
        offset_value_row.addWidget(QLabel("Смещение:"))
        offset_value_row.addWidget(self.depth_offset_all_spin)
        self.depth_offset_all_radio = QRadioButton("Для всех")
        self.depth_offset_separate_radio = QRadioButton("Раздельно")
        self.depth_offset_all_radio.setChecked(True)
        self.depth_offset_separate_radio.toggled.connect(self.on_depth_offset_mode_toggled)
        offset_layout = QVBoxLayout()
        offset_layout.addLayout(offset_value_row)
        offset_layout.addWidget(self.depth_offset_all_radio)
        offset_layout.addWidget(self.depth_offset_separate_radio)
        offset_group.setLayout(offset_layout)

        self.summary_panel = QGroupBox("Сводка")
        depth_box = QGroupBox("Глубины")
        self.max_depth_label = QLabel("—")
        self.min_depth_label = QLabel("—")
        self.avg_depth_label = QLabel("—")
        depth_form = QFormLayout()
        depth_form.addRow("Макс.:", self.max_depth_label)
        depth_form.addRow("Мин.:", self.min_depth_label)
        depth_form.addRow("Средняя:", self.avg_depth_label)
        depth_box.setLayout(depth_form)

        date_box = QGroupBox("Дата")
        self.start_label = QLabel("—")
        self.start_label.setWordWrap(True)
        self.end_label = QLabel("—")
        self.end_label.setWordWrap(True)
        self.duration_label = QLabel("—")
        self.duration_label.setWordWrap(True)
        date_form = QFormLayout()
        date_form.addRow("Начало:", self.start_label)
        date_form.addRow("Конец:", self.end_label)
        date_form.addRow("Продолжительность:", self.duration_label)
        date_box.setLayout(date_form)

        summary_layout = QVBoxLayout()
        summary_layout.addWidget(depth_box)
        summary_layout.addWidget(date_box)
        summary_layout.addStretch(1)
        self.summary_panel.setLayout(summary_layout)

        self.basemap_combo = QComboBox()
        self.basemap_combo.addItems(BASEMAPS.keys())
        self.basemap_combo.currentTextChanged.connect(self.redraw_map)
        self.depth_check = QCheckBox("Глубина")
        self.depth_check.setChecked(True)
        self.depth_check.toggled.connect(self.redraw_map)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Фотоподложка:"))
        row3.addWidget(self.basemap_combo, 1)
        row3.addWidget(self.depth_check)

        left_col = QVBoxLayout()
        left_col.addWidget(crop_group)
        left_col.addWidget(offset_group)
        left_col.addWidget(self.summary_panel)
        left_col.addStretch(1)
        left_widget = QWidget()
        left_widget.setLayout(left_col)
        left_widget.setFixedWidth(230)

        maps_row = QHBoxLayout()
        maps_row.addWidget(left_widget)
        maps_row.addWidget(self.map_view, 1)

        layout = QVBoxLayout()
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row_offset)
        layout.addLayout(maps_row, 1)
        layout.addLayout(row3)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Выберите файл эхограммы")
        QTimer.singleShot(500, lambda: self.check_for_updates(silent=True))

    def pick_sl2(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Файл эхограммы", self.last_dir(), "Lowrance (*.sl2)")
        if path:
            self.remember_dir(path)
            self.file_data = []
            self.depth_offset_per_file = {}
            self.load_file(path, ask_more=False)

    def view_echogram(self):
        path = self.sl2_edit.text()
        if not path or ";" in path:
            QMessageBox.information(self, "Просмотр эхограммы",
                                     "Сначала выберите один файл эхограммы (кнопка «Обзор…»).")
            return
        axis_mode = self.settings.value("timeline_axis", "time")
        dlg = EchogramViewDialog(self, path, axis_mode)
        dlg.exec()

    def pick_gnss(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "GNSS-трек", self.last_dir(), "GNSS-трек (*.nmea *.pos *.csv);;Все файлы (*)")
        if path:
            self.remember_dir(path)
            self.gnss_paths = [path]
            self.gnss_edit.setText(path)
            self.update_offset_enabled()

    def pick_multi_gnss(self):
        self.gnss_paths = []
        self._pick_next_gnss_file()

    def _pick_next_gnss_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "GNSS-трек", self.last_dir(), "GNSS-трек (*.nmea *.pos *.csv);;Все файлы (*)")
        if not path:
            return
        self.remember_dir(path)
        self.gnss_paths.append(path)
        self.gnss_edit.setText("; ".join(self.gnss_paths))
        self.update_offset_enabled()
        reply = QMessageBox.question(
            self, "Мультитрек", "Загрузить ещё один GNSS-трек?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._pick_next_gnss_file()

    def close_echograms(self):
        self.file_data = []
        self.depth_offset_per_file = {}
        self.points = None
        self.sl2_edit.setText("")
        self.update_offset_enabled()
        self.combine_and_render()
        self.statusBar().showMessage("Эхограммы закрыты")

    def close_gnss(self):
        self.gnss_paths = []
        self.gnss_edit.setText("")
        self.update_offset_enabled()
        self.statusBar().showMessage("GNSS-трек закрыт")

    def pick_multi_sl2(self):
        self.file_data = []
        self.depth_offset_per_file = {}
        self._pick_next_multi_file()

    def _pick_next_multi_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Файл эхограммы", self.last_dir(), "Lowrance (*.sl2)")
        if not path:
            return
        self.remember_dir(path)
        self.load_file(path, ask_more=True)

    def last_dir(self):
        return self.settings.value("last_dir", "")

    def remember_dir(self, file_path):
        self.settings.setValue("last_dir", os.path.dirname(file_path))

    def update_offset_enabled(self):
        self.offset_btn.setEnabled(bool(self.sl2_edit.text() and self.gnss_paths))

    def load_file(self, path, ask_more):
        self.statusBar().showMessage("Чтение файла…")
        run_async(lambda: read_echogram_file(path),
                  on_finished=lambda r: self.on_file_loaded(r, ask_more),
                  on_error=self.on_echogram_error)

    def on_file_loaded(self, record, ask_more):
        self.file_data.append(record)
        self.depth_offset_per_file.setdefault(record["path"], 0.0)
        self.sl2_edit.setText("; ".join(f["path"] for f in self.file_data))
        self.update_offset_enabled()
        self.combine_and_render()
        if ask_more:
            reply = QMessageBox.question(
                self, "Мультиэхограмма", "Загрузить ещё одну эхограмму?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self._pick_next_multi_file()

    def crop_mask(self, f):
        if self.crop_mode == "time":
            axis = np.array(f["t_rel"])
            start_cut, end_cut = self.crop_start_time_spin.value(), self.crop_end_time_spin.value()
        else:
            axis = np.array(f["dist"])
            start_cut, end_cut = self.crop_start_dist_spin.value(), self.crop_end_dist_spin.value()
        if len(axis) == 0:
            return axis.astype(bool)
        return (axis - axis[0] >= start_cut) & (axis[-1] - axis >= end_cut)

    def combine_and_render(self):
        lat_all, lon_all, val_all, t_all, dist_all = [], [], [], [], []
        t_offset = dist_offset = 0.0
        for f in self.file_data:
            keep = self.crop_mask(f)
            off_cm = (self.depth_offset_per_file.get(f["path"], 0.0)
                      if self.depth_offset_mode == "separate"
                      else self.depth_offset_all_spin.value())
            off_m = off_cm / 100.0
            lat_all.extend(np.asarray(f["lat"])[keep].tolist())
            lon_all.extend(np.asarray(f["lon"])[keep].tolist())
            val_all.extend((np.asarray(f["depth"])[keep] + off_m).tolist())
            t_all.extend((np.asarray(f["t_rel"])[keep] + t_offset).tolist())
            dist_all.extend((np.asarray(f["dist"])[keep] + dist_offset).tolist())
            # непрерывная ось при нескольких файлах — сдвигаем следующий на длину текущего
            t_offset += f["t_rel"][-1] if f["t_rel"] else 0.0
            dist_offset += f["dist"][-1] if f["dist"] else 0.0

        n = len(self.file_data)
        if n == 0:
            label, start, end, duration_s = "", None, None, None
        elif n == 1:
            label = "глубина, м (встроенный GPS Lowrance)"
            start = self.file_data[0]["start"]
            end = self.file_data[0]["end"]
            duration_s = self.file_data[0]["duration_s"]
        else:
            label = f"глубина, м, {n} эхограмм (встроенный GPS Lowrance)"
            start, end, duration_s = None, None, None

        self.points = dict(lat=lat_all, lon=lon_all, value=val_all, t_rel=t_all, dist=dist_all,
                            label=label, start=start, end=end, duration_s=duration_s)
        if lat_all:
            self.statusBar().showMessage(f"Готово: {len(lat_all)} точек, {label}")
        self.update_summary()
        self.redraw_map()

    def on_crop_mode_toggled(self, checked):
        self.crop_mode = "time" if checked else "distance"
        self.update_crop_enabled()
        if self.file_data:
            self.combine_and_render()

    def update_crop_enabled(self):
        is_time = self.crop_mode == "time"
        self.crop_start_time_spin.setEnabled(is_time)
        self.crop_end_time_spin.setEnabled(is_time)
        self.crop_start_dist_spin.setEnabled(not is_time)
        self.crop_end_dist_spin.setEnabled(not is_time)

    def on_crop_value_changed(self, _value):
        if self.file_data:
            self.combine_and_render()

    def on_depth_offset_all_changed(self, _value):
        if self.depth_offset_mode == "all" and self.file_data:
            self.combine_and_render()

    def on_depth_offset_mode_toggled(self, checked):
        self.depth_offset_all_spin.setEnabled(not checked)
        if checked:
            self.open_depth_offset_dialog()
        else:
            self.depth_offset_mode = "all"
            if self.file_data:
                self.combine_and_render()

    def open_depth_offset_dialog(self):
        if not self.file_data:
            self.depth_offset_mode = "separate"
            return
        dlg = DepthOffsetDialog(self, self.file_data, self.depth_offset_per_file)
        if dlg.exec():
            self.depth_offset_per_file = dlg.values()
            self.depth_offset_mode = "separate"
            self.combine_and_render()
        else:
            self.depth_offset_all_radio.setChecked(True)

    def update_summary(self):
        points = self.points or {}
        val = points.get("value", [])
        if val:
            self.max_depth_label.setText(f"{max(val):.2f} м")
            self.min_depth_label.setText(f"{min(val):.2f} м")
            self.avg_depth_label.setText(f"{sum(val) / len(val):.2f} м")
        else:
            self.max_depth_label.setText("—")
            self.min_depth_label.setText("—")
            self.avg_depth_label.setText("—")

        start, end, duration_s = points.get("start"), points.get("end"), points.get("duration_s")
        if start and end and duration_s is not None:
            self.start_label.setText(datetime.fromisoformat(start).strftime("%d.%m.%Y %H:%M:%S"))
            self.end_label.setText(datetime.fromisoformat(end).strftime("%d.%m.%Y %H:%M:%S"))
            self.duration_label.setText(str(timedelta(seconds=int(duration_s))))
        else:
            self.start_label.setText("—")
            self.end_label.setText("—")
            self.duration_label.setText("—")

    def on_echogram_error(self, message):
        self.statusBar().showMessage("Ошибка чтения файла")
        QMessageBox.critical(self, "Ошибка чтения файла", message)

    def redraw_map(self):
        points = self.points or dict(lat=[], lon=[], value=[], t_rel=[], dist=[])
        manual = self.settings.value("depth_mode", "auto") == "manual"
        vmin = float(self.settings.value("depth_min", 0.0)) if manual else None
        vmax = float(self.settings.value("depth_max", 10.0)) if manual else None
        steps = int(self.settings.value("depth_steps", 32))
        axis_mode = self.settings.value("timeline_axis", "time")
        html = build_map_html(self.basemap_combo.currentText(), points, self.depth_check.isChecked(),
                               vmin=vmin, vmax=vmax, steps=steps, axis_mode=axis_mode)
        load_html(self.map_view, html, "main")

    def open_settings(self):
        dlg = SettingsDialog(self, self.settings)
        if dlg.exec():
            self.redraw_map()

    def check_for_updates(self, silent=False):
        run_async(fetch_latest_release,
                  on_finished=lambda r: self.on_update_checked(r, silent),
                  on_error=lambda m: self.on_update_error(m, silent))

    def on_update_checked(self, result, silent):
        if result["has_update"]:
            reply = QMessageBox.question(
                self, "Доступно обновление",
                f"Вышла новая версия {result['tag']} (у вас {APP_VERSION}).\n\n"
                f"{result['notes'][:500]}\n\nОткрыть страницу релиза?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                QDesktopServices.openUrl(QUrl(result["url"]))
        elif not silent:
            QMessageBox.information(self, "Обновления", "У вас установлена последняя версия.")

    def on_update_error(self, message, silent):
        if not silent:
            QMessageBox.warning(self, "Обновления", f"Не удалось проверить обновления:\n{message}")

    def calc_offset(self):
        self.offset_btn.setEnabled(False)
        self.statusBar().showMessage("Вычисление смещения…")
        run_async(lambda: compute_time_offset(self.sl2_edit.text(), self.gnss_paths),
                  on_finished=self.on_offset_finished, on_error=self.on_offset_error)

    def on_offset_finished(self, result):
        self.statusBar().showMessage(f"Смещение: {result['offset_s']:+.3f} с")
        self.offset_btn.setEnabled(True)
        dlg = OffsetDialog(self, self.basemap_combo.currentText(), result)
        dlg.exec()

    def on_offset_error(self, message):
        self.statusBar().showMessage("Ошибка вычисления смещения")
        self.offset_btn.setEnabled(True)
        QMessageBox.critical(self, "Ошибка вычисления смещения", message)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
