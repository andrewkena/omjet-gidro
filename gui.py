#!/usr/bin/env python3
"""
Главное окно sl2sync: выбор файла эхограммы, карта с точками (встроенный GPS
Lowrance) поверх выбранной онлайн-подложки, расчёт временного смещения
относительно GNSS-трека со сравнением треков до/после синхронизации.
"""
import csv
import itertools
import json
import os
import re
import sys
import tempfile
import threading
import urllib.request
from datetime import datetime, timedelta

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from PySide6.QtCore import QObject, QRect, QSettings, QSize, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QApplication, QCheckBox, QColorDialog, QComboBox, QDialog,
                                QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                                QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMenu,
                                QMainWindow, QMessageBox, QPushButton, QRadioButton,
                                QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget)

from sl2sync import (build_parser, estimate_time_model, gnss_motion, make_proj,
                      ping_positions, read_gnss, read_sl2)

APP_VERSION = "0.1.1"
GITHUB_REPO = "andrewkena/omjet-gidro"
ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "gidro.ico")

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


def help_icon(tooltip):
    """Значок «?» с описанием принимаемых файлов во всплывающей подсказке —
    ставится рядом с полями выбора файла, чтобы не загромождать интерфейс
    текстом, но дать ответ на вопрос «а что сюда можно загрузить»."""
    lbl = QLabel("?")
    lbl.setToolTip(tooltip)
    lbl.setFixedSize(18, 18)
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet(
        "QLabel { background: #555; color: #eee; border-radius: 9px; font-weight: bold; }")
    lbl.setCursor(Qt.CursorShape.WhatsThisCursor)
    return lbl


def label_with_help(text, tooltip):
    """Подпись поля/строки отчёта + значок «?» с пояснением рядом — для мест,
    где подписи в форме (QFormLayout.addRow принимает виджет вместо строки)."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(QLabel(text))
    lay.addWidget(help_icon(tooltip))
    return w


SL2_FILE_HELP = "Запись эхолота Lowrance — файл .sl2."
GNSS_FILE_HELP = (
    "GNSS-трек, один из форматов:\n"
    "• NMEA-лог (.nmea) — строки GGA + RMC/ZDA\n"
    "• RTKLIB .pos — результат постобработки PPK (lat/lon/height, не ECEF/ENU)\n"
    "• CSV — колонки time, lat, lon (опционально h, q)\n"
    "• UBX (.ubx) — бинарный поток u-blox: NAV-PVT, либо NAV-POSLLH +\n"
    "  NAV-STATUS + NAV-TIMEUTC.\n"
    "Для точного RTK лучше сначала обработать пару Base+Rover .ubx через\n"
    "RTKLIB в .pos — прямое чтение .ubx даёт только качество фикса самого\n"
    "приёмника (обычно одиночный, без поправок в реальном времени)."
)
WATERLINE_FILE_HELP = (
    "Точки уреза воды — CSV с колонками lat/lon (разделитель определяется\n"
    "автоматически) либо обычный текст: на каждой строке два числа lat lon."
)


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


OMSK_CENTER = [54.9885, 73.3242]


def build_map_html(basemap, points, color_by_depth=True, vmin=None, vmax=None, steps=32,
                    axis_mode="time", legend_title="Глубина, м", tooltip_suffix=" м",
                    depth_vals=None, speed_vals=None, show_endpoints=True,
                    show_speed_track=False, track_width=3, speed_glow_width=12,
                    gnss_track=None):
    tile = BASEMAPS[basemap]
    lat, lon, val = points["lat"], points["lon"], points["value"]
    axis_key = "dist" if axis_mode == "distance" else "t_rel"
    axis_vals = points.get(axis_key) or list(range(len(val)))
    if lat:
        center = [sum(lat) / len(lat), sum(lon) / len(lon)]
        zoom = 15
    else:
        # До загрузки трека — открываем карту на Омске, а не в океане у [0,0].
        center = OMSK_CENTER
        zoom = 12
    if vmin is None or vmax is None:
        vmin, vmax = (min(val), max(val)) if val else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    depth_vals = list(depth_vals) if depth_vals is not None else list(val)
    max_depth_idx = depth_vals.index(max(depth_vals)) if depth_vals else -1
    max_depth_val = float(depth_vals[max_depth_idx]) if max_depth_idx >= 0 else 0.0
    speed_vals = list(speed_vals) if speed_vals is not None else []
    max_speed_idx = speed_vals.index(max(speed_vals)) if speed_vals else -1
    max_speed_val = float(speed_vals[max_speed_idx]) if max_speed_idx >= 0 else 0.0
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
.depth-legend .bar-wrap{{position:relative;width:12px;height:120px}}
.depth-legend .bar{{width:12px;height:120px;
               background:linear-gradient(to top, rgb(128,0,0), rgb(255,0,0), rgb(255,255,0),
               rgb(0,255,255), rgb(0,0,255), rgb(0,0,143))}}
.depth-legend .tick{{position:absolute;left:0;width:12px;height:1px;
               background:rgba(0,0,0,.55)}}
.depth-legend .scale{{display:flex;flex-direction:column;justify-content:space-between;
               height:120px;text-align:center}}
.depth-legend .row{{display:flex;align-items:stretch}}
.depth-cursor-tip{{background:#222;color:#fff;border:none;font:12px sans-serif}}
.track-flag{{background:transparent;border:none}}
.max-depth-label,.max-speed-label{{background:transparent;border:none}}
.max-depth-label span,.max-speed-label span{{background:rgba(0,0,0,.65);color:#fff;padding:1px 4px;
               border-radius:3px;font:11px sans-serif;white-space:nowrap}}</style>
</head><body>
<div id="map"></div>
<canvas id="timeline"></canvas>
<script>
var map = L.map('map', {{preferCanvas: true, attributionControl: false}}).setView([{center[0]}, {center[1]}], {zoom});
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
var showSpeedTrack = {"true" if show_speed_track else "false"};
// В режиме «Скорость на треке» точки, окрашенные по глубине/твёрдости, не
// показываем вовсе — только скорость, без смешения с другой раскраской. Слой
// всё равно нужен (getBounds() для авто-масштабирования карты), просто не
// добавляем его на карту.
var layer = L.geoJSON(data, {{
  pointToLayer: function (f, latlng) {{
    var m = L.circleMarker(latlng, {{radius: 3, weight: 0, fillOpacity: 0.85,
                                      fillColor: color(f.properties.v), color: color(f.properties.v)}});
    m.bindTooltip(f.properties.v.toFixed(2) + {json.dumps(tooltip_suffix)}, {{sticky: true}});
    m.on('mouseover', function () {{ showAtIndex(f.properties.i); }});
    return m;
  }}
}});
if (!showSpeedTrack) {{ layer.addTo(map); }}

var gnssTrackPts = {json.dumps([[la, lo] for la, lo in zip((gnss_track or {}).get("lat", []),
                                                             (gnss_track or {}).get("lon", []))])};
if (gnssTrackPts.length > 1) {{
  L.polyline(gnssTrackPts, {{color: '#ff0000', weight: 1.5, opacity: 0.85,
                             interactive: false}}).addTo(map);
}}

var speedVals = {json.dumps(speed_vals)};
var trackLineWidth = {int(track_width)};
var speedGlowWidth = {int(speed_glow_width)};
var speedMin = 0, speedMax = 1;
if (speedVals.length) {{
  speedMin = Math.min.apply(null, speedVals);
  speedMax = Math.max.apply(null, speedVals);
}}
if (speedMax <= speedMin) {{ speedMax = speedMin + 1e-6; }}
function speedColor(v) {{
  var t = Math.max(0, Math.min(1, (v - speedMin) / (speedMax - speedMin)));
  return jetColor(t);
}}
// Отдельный pane с z-index ниже слоя точек трека (overlayPane = 400) — свечение
// скорости всегда рисуется под точками, независимо от порядка добавления слоёв,
// и выходит за их пределы (перекрывает трек снизу), а не поверх них.
var speedPane = map.createPane('speedGlowPane');
speedPane.style.zIndex = 350;
speedPane.style.pointerEvents = 'none';

var speedTrackLayer = L.layerGroup();
if (speedVals.length === pts.length && pts.length > 1) {{
  for (var si = 0; si < pts.length - 1; si++) {{
    var segCol = speedColor((speedVals[si] + speedVals[si + 1]) / 2);
    var segCoords = [[pts[si].lat, pts[si].lon], [pts[si + 1].lat, pts[si + 1].lon]];
    L.polyline(segCoords, {{pane: 'speedGlowPane', color: segCol, weight: speedGlowWidth,
                             opacity: 0.35, interactive: false}}).addTo(speedTrackLayer);
    L.polyline(segCoords, {{pane: 'speedGlowPane', color: segCol, weight: trackLineWidth,
                             opacity: 0.7, interactive: false}}).addTo(speedTrackLayer);
  }}
}}

var speedLegend = L.control({{position: 'bottomright'}});
speedLegend.onAdd = function () {{
  var div = L.DomUtil.create('div', 'depth-legend');
  var nTicks = 5;
  // Полный градиент по тем же опорным точкам, что и jetColor (а не только
  // min/max) — иначе полоса легенды не показывает голубой/жёлтый переход,
  // который реально виден на треке для промежуточных скоростей.
  var speedStops = jetStops.map(function (s) {{
    return 'rgb(' + s[1][0] + ',' + s[1][1] + ',' + s[1][2] + ')';
  }}).join(', ');
  var barHtml = '<div class="bar-wrap"><div class="bar" style="background:' +
    'linear-gradient(to top, ' + speedStops + ')"></div>';
  var scaleHtml = '<div class="scale">';
  for (var k = 0; k <= nTicks; k++) {{
    barHtml += '<div class="tick" style="top:' + (k / nTicks * 100) + '%"></div>';
    scaleHtml += '<span>' + (speedMax - (speedMax - speedMin) * (k / nTicks)).toFixed(2) + '</span>';
  }}
  barHtml += '</div>';
  scaleHtml += '</div>';
  div.innerHTML = '<div>Скорость, м/с</div><div class="row">' + barHtml + scaleHtml + '</div>';
  return div;
}};
var speedLegendAdded = false;
function setSpeedTrackVisible(v) {{
  if (v) {{
    if (!map.hasLayer(speedTrackLayer)) {{ map.addLayer(speedTrackLayer); }}
    if (!speedLegendAdded) {{ speedLegend.addTo(map); speedLegendAdded = true; }}
  }} else {{
    if (map.hasLayer(speedTrackLayer)) {{ map.removeLayer(speedTrackLayer); }}
    if (speedLegendAdded) {{ speedLegend.remove(); speedLegendAdded = false; }}
  }}
}}
setSpeedTrackVisible(showSpeedTrack);

var showEndpoints = {"true" if show_endpoints else "false"};
var maxDepthIdx = {max_depth_idx};
var maxDepthVal = {max_depth_val};
var maxSpeedIdx = {max_speed_idx};
var maxSpeedVal = {max_speed_val};
function flagIcon(color) {{
  return L.divIcon({{
    className: 'track-flag', iconSize: [18, 22], iconAnchor: [2, 20],
    html: '<svg width="18" height="22" viewBox="0 0 18 22">' +
          '<line x1="2" y1="2" x2="2" y2="20" stroke="#333" stroke-width="2"/>' +
          '<path d="M2,2 L16,6 L2,10 Z" fill="' + color + '" stroke="#000" stroke-width="0.5"/>' +
          '</svg>'
  }});
}}
if (showEndpoints && pts.length) {{
  L.marker([pts[0].lat, pts[0].lon], {{icon: flagIcon('#2ecc40'), interactive: false}}).addTo(map);
  var lastP = pts[pts.length - 1];
  L.marker([lastP.lat, lastP.lon], {{icon: flagIcon('#e74c3c'), interactive: false}}).addTo(map);
  if (maxDepthIdx >= 0 && !showSpeedTrack) {{
    var mp = pts[maxDepthIdx];
    L.circleMarker([mp.lat, mp.lon], {{radius: 5, weight: 1, color: '#fff',
                                        fillColor: '#000', fillOpacity: 1, interactive: false}}).addTo(map);
    L.marker([mp.lat, mp.lon], {{
      icon: L.divIcon({{className: 'max-depth-label', iconSize: null, iconAnchor: [-8, 6],
                        html: '<span>' + maxDepthVal.toFixed(2) + ' м</span>'}}),
      interactive: false
    }}).addTo(map);
  }}
  if (maxSpeedIdx >= 0) {{
    var sp = pts[maxSpeedIdx];
    L.circleMarker([sp.lat, sp.lon], {{radius: 5, weight: 1, color: '#fff',
                                        fillColor: '#e74c3c', fillOpacity: 1, interactive: false}}).addTo(map);
    L.marker([sp.lat, sp.lon], {{
      icon: L.divIcon({{className: 'max-speed-label', iconSize: null, iconAnchor: [-8, 6],
                        html: '<span>' + maxSpeedVal.toFixed(2) + ' м/с</span>'}}),
      interactive: false
    }}).addTo(map);
  }}
}}

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
  cursorMarker.setTooltipContent(p.v.toFixed(2) + {json.dumps(tooltip_suffix)});
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
  if (colorByDepth && !showSpeedTrack) {{
    var legend = L.control({{position: 'bottomright'}});
    legend.onAdd = function () {{
      var div = L.DomUtil.create('div', 'depth-legend');
      var nTicks = 5;
      var barHtml = '<div class="bar-wrap"><div class="bar"></div>';
      var scaleHtml = '<div class="scale">';
      for (var k = 0; k <= nTicks; k++) {{
        barHtml += '<div class="tick" style="top:' + (k / nTicks * 100) + '%"></div>';
        scaleHtml += '<span>' + (vmax - (vmax - vmin) * (k / nTicks)).toFixed(1) + '</span>';
      }}
      barHtml += '</div>';
      scaleHtml += '</div>';
      div.innerHTML = '<div>' + {json.dumps(legend_title)} + '</div><div class="row">' +
        barHtml + scaleHtml + '</div>';
      return div;
    }};
    legend.addTo(map);
  }}
}}
</script>
</body></html>"""


def build_tracks_html(basemap, tracks, depth_vmin=None, depth_vmax=None, sync=False):
    """tracks: список dict(name, color, lat, lon[, depth]) — трек с ключом
    depth рисуется точками, окрашенными по глубине (та же радужная палитра,
    что на основной карте: минимум — красный, максимум — синий), остальные —
    обычной сплошной линией цвета color (например, GNSS-трек — глубины нет).

    sync=True подключает QWebChannel (см. TrackCompareBridge в OffsetDialog):
    движение карты (пан/зум) и наведение на точку трека с глубиной
    транслируются на вторую карту через Python — сами карты это две разные
    страницы в разных QWebEngineView, прямой связи между их JS нет."""
    tile = BASEMAPS[basemap]
    all_lat = [la for t in tracks for la in t["lat"]]
    all_lon = [lo for t in tracks for lo in t["lon"]]
    center = [sum(all_lat) / len(all_lat), sum(all_lon) / len(all_lon)] if all_lat else [0, 0]

    depth_vals = [d for t in tracks for d in (t.get("depth") or [])]
    if depth_vmin is None:
        depth_vmin = min(depth_vals) if depth_vals else 0.0
    if depth_vmax is None:
        depth_vmax = max(depth_vals) if depth_vals else 1.0
    if depth_vmax <= depth_vmin:
        depth_vmax = depth_vmin + 1e-6

    lines = []
    for t in tracks:
        coords = list(zip(t["lat"], t["lon"]))
        depth = t.get("depth")
        if depth:
            ping_idx = t.get("ping_idx") or list(range(len(coords)))
            lines.append(f"drawDepthTrack({json.dumps(coords)}, {json.dumps(depth)}, "
                          f"{json.dumps(ping_idx)}, {json.dumps(t['name'])});")
        else:
            lines.append(
                f"L.polyline({json.dumps(coords)}, {{color: '{t['color']}', weight: 4}})"
                f".bindTooltip({json.dumps(t['name'])}).addTo(map);")
    bounds = list(zip(all_lat, all_lon))
    channel_script = ('<script src="qrc:///qtwebchannel/qwebchannel.js"></script>'
                       if sync else '')
    channel_init = ('new QWebChannel(qt.webChannelTransport, '
                     'function (channel) { bridge = channel.objects.bridge; });'
                     if sync else '')
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
{channel_script}
<style>html,body,#map{{height:100%;margin:0}}
.hover-tip{{background:#222;color:#fff;border:none;font:12px sans-serif}}</style>
</head><body>
<div id="map"></div>
<script>
var map = L.map('map', {{preferCanvas: true, attributionControl: false}}).setView([{center[0]}, {center[1]}], 15);
L.tileLayer('{tile["url"]}', {{maxZoom: {tile["max_zoom"]}}}).addTo(map);
var depthVmin = {depth_vmin}, depthVmax = {depth_vmax};
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
function depthColor(v) {{
  var t = Math.max(0, Math.min(1, (v - depthVmin) / (depthVmax - depthVmin)));
  return jetColor(1 - t);
}}

var bridge = null;
var suppressMove = false;
{channel_init}

var cursorMarker = L.circleMarker([0, 0], {{radius: 8, color: '#fff', weight: 2,
                                             fillColor: '#ffd400', fillOpacity: 1}});
cursorMarker.bindTooltip('', {{permanent: true, direction: 'top', className: 'hover-tip'}});
var depthTrackCoords = [], depthTrackVals = [], depthTrackPingIdx = [];

function showCursor(idx) {{
  if (idx < 0 || idx >= depthTrackCoords.length) {{
    if (cursorMarker._map) {{ map.removeLayer(cursorMarker); }}
    return;
  }}
  var p = depthTrackCoords[idx];
  if (!cursorMarker._map) {{ cursorMarker.addTo(map); }}
  cursorMarker.setLatLng(p);
  cursorMarker.setTooltipContent(depthTrackVals[idx].toFixed(2) + ' м');
  cursorMarker.openTooltip();
}}

// depthTrackPingIdx отсортирован по возрастанию (порядок пингов в записи) —
// двоичный поиск ближайшего к target номера пинга.
function findClosestByPingIdx(target) {{
  var arr = depthTrackPingIdx;
  if (!arr.length) {{ return -1; }}
  var lo = 0, hi = arr.length - 1;
  while (lo < hi) {{
    var mid = (lo + hi) >> 1;
    if (arr[mid] < target) {{ lo = mid + 1; }} else {{ hi = mid; }}
  }}
  if (lo > 0 && Math.abs(arr[lo - 1] - target) <= Math.abs(arr[lo] - target)) {{ return lo - 1; }}
  return lo;
}}

// Вызывается из Python — эхо наведения со второй карты. pingIdx — настоящий
// номер пинга (не позиция в массиве этой карты — маски "до"/"после" разные,
// см. compute_time_offset в gui.py), поэтому ищем ближайший имеющийся у себя.
function remoteHighlight(pingIdx) {{
  showCursor(pingIdx < 0 ? -1 : findClosestByPingIdx(pingIdx));
}}

// Вызывается из Python — эхо пана/зума второй карты; suppressMove не даёт уйти в петлю.
function syncView(lat, lng, zoom) {{
  suppressMove = true;
  map.setView([lat, lng], zoom);
  suppressMove = false;
}}

map.on('moveend', function () {{
  if (!suppressMove && bridge) {{
    var c = map.getCenter();
    bridge.viewChanged(c.lat, c.lng, map.getZoom());
  }}
}});
map.on('mouseout', function () {{
  showCursor(-1);
  if (bridge) {{ bridge.hoverEnd(); }}
}});

function drawDepthTrack(coords, depth, pingIdx, name) {{
  depthTrackCoords = coords;
  depthTrackVals = depth;
  depthTrackPingIdx = pingIdx;
  var group = L.layerGroup().addTo(map);
  for (let i = 0; i < coords.length; i++) {{
    L.circleMarker(coords[i], {{radius: 4, weight: 0, fillOpacity: 0.85,
                                 fillColor: depthColor(depth[i]), color: depthColor(depth[i])}})
      .on('mouseover', function () {{
        showCursor(i);
        if (bridge) {{ bridge.hoverPoint(pingIdx[i]); }}
      }})
      .addTo(group);
  }}
}}
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


def estimate_bottom_hardness_ping(record, depth_m, range_m=None):
    """Грубая, неоткалиброванная оценка твёрдости дна по ширине первого пика
    (резче — твёрже) и наличию второго эха на удвоенной глубине (сигнал
    поверхность→дно→поверхность→дно; заметен обычно только на твёрдом дне).
    Формат сырых байт эхограммы не задокументирован и не проверен на реальном
    железе (см. read_echogram_waterfall) — это ориентировочный индекс, а не
    калиброванное измерение, как у специализированных систем (RoxAnn, QTC).

    range_m — окно показа сонара для этого пинга (см. read_sl2, поле range_ft):
    Lowrance меняет его автоматически ступенями, поэтому байт ≠ постоянная доля
    метра между пингами — без пересчёта ширины пика в метры через range_m
    резкость на бо́льшем диапазоне (меньше байт на метр) систематически
    получалась «мягче» независимо от реальной твёрдости дна. Также используется
    (вместе с depth_m) для поиска самого пика дна: первые байты пинга — это
    выброс от импульса излучения (плато почти одинаковых значений у
    поверхности, обычно выше настоящего эха от дна — проверено на реальных
    данных) и глобальный максимум по всему столбцу почти всегда попадает
    именно в этот выброс, а не в дно. Ищем пик в окне вокруг уже известной
    глубины (из штатного трекера эхолота, depth_m), а не по всему столбцу."""
    if not record or depth_m is None or depth_m <= 0:
        return None
    arr = np.frombuffer(record, dtype=np.uint8).astype(np.float32)
    n = len(arr)
    if n < 4:
        return None
    if range_m and range_m > 0:
        expected = depth_m / range_m * n
        margin = max(10.0, 0.25 * expected)
        lo, hi = max(0, int(expected - margin)), min(n, int(expected + margin) + 1)
    else:
        # Без range_m окно вокруг ожидаемого байта не посчитать — просто
        # отбрасываем небольшой начальный участок с выбросом от импульса.
        lo, hi = min(n - 1, max(4, int(n * 0.03))), n
    if hi - lo < 4:
        return None
    i_peak = lo + int(np.argmax(arr[lo:hi]))
    peak_val = arr[i_peak]
    if peak_val < 20:
        return None
    half = peak_val / 2.0
    left = i_peak
    while left > 0 and arr[left] > half:
        left -= 1
    right = i_peak
    while right < n - 1 and arr[right] > half:
        right += 1
    width = max(1, right - left)
    if range_m:
        width = width * (range_m / n)
    sharpness = 1.0 / width

    second_echo = 0.0
    i2 = 2 * i_peak
    if i2 + 2 < n:
        second_echo = float(np.max(arr[max(0, i2 - 2):i2 + 3])) / 255.0
    return sharpness, second_echo


def compute_hardness_index(records, depth_m, range_m=None):
    n = len(records)
    sharpness = np.full(n, np.nan)
    second = np.zeros(n)
    for i in range(n):
        r = range_m[i] if range_m is not None else None
        est = estimate_bottom_hardness_ping(records[i], depth_m[i], r)
        if est is not None:
            sharpness[i], second[i] = est
    finite = sharpness[np.isfinite(sharpness)]
    if len(finite) > 1:
        lo, hi = np.nanpercentile(finite, [5, 95])
        sharp_n = np.clip((sharpness - lo) / max(hi - lo, 1e-9), 0, 1)
    else:
        sharp_n = np.zeros(n)
    sharp_n = np.nan_to_num(sharp_n, nan=0.0)
    return (0.5 * sharp_n + 0.5 * second).tolist()


def read_echogram_file(sl2_path):
    s, info = read_sl2(sl2_path, with_echogram=True)
    m = s["has_gps"] & np.isfinite(s["depth_m"]) & (s["depth_m"] > 0)
    lat, lon = s["lat"][m], s["lon"][m]
    t_rel = s["t_rel"][m]
    depth = s["depth_m"][m]
    speed = s["speed_ms"][m]
    range_m = s["range_m"][m]
    records = [s["echogram"][i] for i in np.where(m)[0]]
    hardness = compute_hardness_index(records, depth.tolist(), range_m.tolist())
    duration_s = info["duration_s"]
    if info.get("start_epoch_utc") is not None:
        # Настоящее время начала записи из самого файла (поле "unix_s" первого
        # кадра, см. read_sl2 в sl2sync.py) — надёжнее mtime, который зависит от
        # того, как файл попал на диск (копирование/перенос с карты памяти может
        # исказить дату, что и произошло на реальных файлах пользователя).
        start_dt = datetime.utcfromtimestamp(info["start_epoch_utc"])
        end_dt = start_dt + timedelta(seconds=duration_s)
    else:
        # Резервный вариант, если поле не распознано (например, повреждён первый
        # кадр) — как раньше, от mtime файла (момент завершения записи) назад.
        end_dt = datetime.utcfromtimestamp(os.path.getmtime(sl2_path))
        start_dt = end_dt - timedelta(seconds=duration_s)
    return dict(path=sl2_path, lat=lat.tolist(), lon=lon.tolist(),
                depth=depth.tolist(), t_rel=t_rel.tolist(), speed=speed.tolist(),
                dist=cumulative_distance(lat, lon), hardness=hardness,
                start=start_dt.isoformat(), end=end_dt.isoformat(), duration_s=duration_s)


def read_echogram_waterfall(sl2_path):
    """Сырые сэмплы эхограммы (амплитуда сонара) по каждому пингу — байты сразу
    после 144-байтного заголовка кадра. Формат байтов документально не описан
    в docs/CONTEXT.md и не проверен на реальном железе (см. открытые вопросы
    там же) — по общепринятому для sl2 предположению это 8-битная амплитуда
    по глубине, старт столбца сверху (поверхность) вниз (дно). range_m — окно
    показа сонара для каждого пинга (см. read_sl2) — нужно для калибровки
    столбцов эхограммы под единый масштаб глубины (build_echogram_image)."""
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
                range_m=s["range_m"][m].tolist(),
                t_rel=s["t_rel"][m].tolist(),
                dist=cumulative_distance(s["lat"][m], s["lon"][m]),
                mixed_freqs=mixed_freqs)


def format_axis_label(v, axis_mode):
    if axis_mode == "distance":
        return f"{v / 1000:.1f} км" if v >= 1000 else f"{v:.0f} м"
    m, sec = divmod(int(round(v)), 60)
    return f"{m}:{sec:02d}"


def build_echogram_image(records, range_m=None):
    """Складываем сырые байты пинга: строка 0 — начало записи каждого пинга
    (поверхность), дальше — по возрастанию байта. Короткие пинги дополняются
    как «нет данных» (валидная маска).

    range_m — окно показа сонара для каждого пинга, метры (см. read_sl2:
    поле range_ft, найдено и проверено на реальных файлах — Lowrance меняет
    его автоматически ступенями, 4/6/10/16/20/30 м на проверенных файлах, и
    оно всегда больше фактической глубины дна). Раньше здесь была попытка
    растянуть столбец под глубину ДНА (depth_m) в предположении диапазон≈
    глубина — предположение было неверным (диапазон ощутимо больше глубины)
    и давало ложные резкие скачки. С правильным полем (range_m) столбцы
    пересчитываются к единому масштабу глубины (общий максимальный диапазон
    в записи) через линейную интерполяцию — это убирает видимые «швы» на
    границах смены диапазона. Без range_m (не передан) — старое поведение:
    байты как есть, без калибровки.

    Возвращает (arr, valid, calibrated_range_m) — третий элемент None, если
    калибровка не выполнялась (range_m не передан или непригоден)."""
    lengths = [len(r) for r in records if r]
    if not lengths:
        return None, None, None
    native_rows = max(lengths)
    n = len(records)
    arr = np.zeros((native_rows, n), dtype=np.uint8)
    valid = np.zeros((native_rows, n), dtype=bool)
    for col, r in enumerate(records):
        if r:
            m = len(r)
            arr[:m, col] = np.frombuffer(r, dtype=np.uint8)
            valid[:m, col] = True

    if range_m is None:
        return arr, valid, None
    range_arr = np.asarray(range_m, dtype=np.float32)
    if len(range_arr) != n:
        return arr, valid, None
    target_range = float(np.nanmax(range_arr)) if n else 0.0
    if not np.isfinite(target_range) or target_range <= 0:
        return arr, valid, None

    target_rows = native_rows
    depths = np.linspace(0.0, target_range, target_rows, dtype=np.float32)
    out_arr = np.zeros((target_rows, n), dtype=np.uint8)
    out_valid = np.zeros((target_rows, n), dtype=bool)
    chunk = 4096  # ограничиваем пиковую память при интерполяции больших записей
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        rng = range_arr[start:end]
        with np.errstate(divide="ignore", invalid="ignore"):
            src = depths[:, None] / rng[None, :] * native_rows
        i0 = np.floor(src).astype(np.int32)
        frac = (src - i0).astype(np.float32)
        bad = (i0 < 0) | (i0 >= native_rows - 1) | ~np.isfinite(src)
        i0c = np.clip(i0, 0, native_rows - 2)
        i1c = i0c + 1
        cols = np.broadcast_to(np.arange(end - start), i0c.shape)
        sub_arr, sub_valid = arr[:, start:end], valid[:, start:end]
        a0 = sub_arr[i0c, cols].astype(np.float32)
        a1 = sub_arr[i1c, cols].astype(np.float32)
        blended = a0 * (1 - frac) + a1 * frac
        v = sub_valid[i0c, cols] & sub_valid[i1c, cols] & ~bad
        out_arr[:, start:end] = np.where(v, blended, 0).astype(np.uint8)
        out_valid[:, start:end] = v
    return out_arr, out_valid, target_range


def block_reduce_mean(arr, valid, row_factor, col_factor):
    """Огрубляет эхограмму усреднением по блокам row_factor×col_factor —
    нужно перед показом при сильном уменьшении масштаба: если просто отдать
    Qt десятки тысяч столбцов на отрисовку в узкую полосу, растяжение/сжатие
    у QPainter (даже со SmoothPixmapTransform — это билинейная выборка, а не
    честное усреднение области) даёт алиасинг — зубчатые вертикальные полосы
    вместо гладкой картины, которая видна при масштабе 1:1 (проверено на
    реальном файле). Усредняем только по валидным сэмплам блока, чтобы
    маска «нет данных» (короткие пинги) не затемняла среднее."""
    if row_factor <= 1 and col_factor <= 1:
        return arr, valid
    row_factor, col_factor = max(1, row_factor), max(1, col_factor)
    rows, cols = arr.shape
    new_rows, new_cols = max(1, rows // row_factor), max(1, cols // col_factor)
    arr = arr[:new_rows * row_factor, :new_cols * col_factor]
    valid = valid[:new_rows * row_factor, :new_cols * col_factor]
    arr = arr.reshape(new_rows, row_factor, new_cols, col_factor)
    valid_r = valid.reshape(new_rows, row_factor, new_cols, col_factor)
    valid_count = valid_r.sum(axis=(1, 3))
    arr_sum = np.where(valid_r, arr, 0).sum(axis=(1, 3), dtype=np.float32)
    out_arr = np.divide(arr_sum, valid_count, out=np.zeros_like(arr_sum),
                         where=valid_count > 0)
    return out_arr.astype(np.uint8), valid_count > 0


def array_to_qimage(arr):
    h, w = arr.shape
    arr = np.ascontiguousarray(arr)
    img = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
    return img.copy()


JET_STOPS = [
    (0.0, (0, 0, 143)), (0.125, (0, 0, 255)), (0.375, (0, 255, 255)),
    (0.625, (255, 255, 0)), (0.875, (255, 0, 0)), (1.0, (128, 0, 0)),
]


def _jet_lut():
    """256-элементная таблица RGB той же палитры, что и точки на карте: слабый
    сигнал — синий, сильный — красный/тёмно-красный."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for j in range(len(JET_STOPS) - 1):
            a0, c0 = JET_STOPS[j]
            a1, c1 = JET_STOPS[j + 1]
            if t <= a1 or j == len(JET_STOPS) - 2:
                f = (t - a0) / (a1 - a0) if a1 > a0 else 0.0
                lut[i] = [c0[k] + f * (c1[k] - c0[k]) for k in range(3)]
                break
    return lut


_JET_LUT = _jet_lut()


def colorize_echogram(arr, valid=None):
    """arr: 2D uint8 (строки — глубина, столбцы — пинги) → цветной QImage
    по амплитуде через ту же jet-палитру, что и раскраска точек на карте.
    valid: булева маска того же размера — False красится чёрным (нет данных),
    а не через палитру (иначе ноль амплитуды путается с тёмно-синим слабым
    сигналом)."""
    rgb = np.ascontiguousarray(_JET_LUT[arr])
    if valid is not None:
        rgb[~valid] = 0
    h, w, _ = rgb.shape
    img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
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


def read_gnss_track_points(gnss_paths):
    """Только lat/lon объединённого GNSS-трека — для показа тонкой красной линией
    на главной карте (отдельно от полного compute_time_offset, который нужен
    только при расчёте смещения)."""
    g = merge_gnss_tracks(gnss_paths)
    return dict(lat=g["lat"].tolist(), lon=g["lon"].tolist())


def compute_isobaths(lat, lon, depth, waterline_points, cell, interval, fill_steps, smooth=0.0):
    """Грид глубин (линейная интерполяция) + линии постоянной глубины и залитые
    диапазоны глубин через matplotlib.contour/contourf. Точки уреза воды
    (нарисованные вручную, глубина 0) подмешиваются к данным эхолота — это
    стандартный приём в батиметрии: сонар не измеряет вплотную к берегу,
    а урез задаёт границу 0 м.

    `smooth` — сигма гауссова размытия грида (в ячейках сетки) перед contour/
    contourf: сглаживает и линии, и заливку одинаково, т.к. обе строятся по
    одному Zm. 0 — без сглаживания. Перед размытием маскированные (вне выпуклой
    оболочки промеров) ячейки временно заполняются ближайшим соседом, иначе
    NaN размывается на соседние ячейки; после размытия исходная маска
    возвращается.

    Упрощение: контуры каждого диапазона заливки отдаются как есть, без
    различения внешних границ и дырок (островов) — для типичной акватории
    без островов внутри снятой площади это не имеет значения."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath

    lat_arr = np.array(list(lat) + [p[0] for p in waterline_points])
    lon_arr = np.array(list(lon) + [p[1] for p in waterline_points])
    depth_arr = np.array(list(depth) + [0.0] * len(waterline_points))

    crs, fwd, inv = make_proj(float(np.median(lon_arr)), float(np.median(lat_arr)))
    E, N = fwd.transform(lon_arr, lat_arr)

    x0, y0 = float(E.min()), float(N.min())
    nx = max(2, int((E.max() - x0) / cell) + 1)
    ny = max(2, int((N.max() - y0) / cell) + 1)
    if nx * ny > 4_000_000:
        raise ValueError("Слишком мелкая ячейка сетки для такой площади — увеличьте шаг")
    xs = x0 + np.arange(nx) * cell
    ys = y0 + np.arange(ny) * cell
    GX, GY = np.meshgrid(xs, ys)
    Z = griddata((E, N), depth_arr, (GX, GY), method="linear")
    if np.all(np.isnan(Z)):
        raise ValueError("Не удалось построить сетку — проверьте данные")
    if smooth and smooth > 0:
        nan_mask = np.isnan(Z)
        Zfill = griddata((E, N), depth_arr, (GX, GY), method="nearest")
        Zs = np.where(nan_mask, Zfill, Z)
        Zs = gaussian_filter(Zs, sigma=float(smooth))
        Z = np.where(nan_mask, np.nan, Zs)

    if len(waterline_points) >= 3:
        # Изобаты не должны заходить на сушу за нарисованный урез воды: замыкаем
        # нарисованную линию в многоугольник (от последней точки к первой) и
        # определяем, какая сторона — вода, по тому, где лежит сам трек (лодка
        # не плавает по суше); ячейки сетки на другой стороне маскируем — тогда
        # contour/contourf там ничего не рисуют, без ручного отреза геометрии.
        wl_E, wl_N = fwd.transform([p[1] for p in waterline_points],
                                    [p[0] for p in waterline_points])
        wl_path = MplPath(np.column_stack([wl_E, wl_N]))
        track_pts = np.column_stack([E[:len(lat)], N[:len(lat)]])
        track_inside = wl_path.contains_points(track_pts).mean() >= 0.5
        grid_inside = wl_path.contains_points(
            np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        water_mask = grid_inside if track_inside else ~grid_inside
        Z = np.where(water_mask, Z, np.nan)

    Zm = np.ma.masked_invalid(Z)
    zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    start = np.ceil(zmin / interval) * interval
    line_levels = np.arange(start, zmax, interval)
    if len(line_levels) == 0:
        raise ValueError("Нет подходящих уровней изобат — измените шаг")

    fig, ax = plt.subplots()
    cs = ax.contour(GX, GY, Zm, levels=line_levels)
    lines = []
    for level, segs in zip(cs.levels, cs.allsegs):
        for seg in segs:
            if len(seg) < 2:
                continue
            lon_c, lat_c = inv.transform(seg[:, 0], seg[:, 1])
            lines.append(dict(level=float(level),
                               coords=[[float(la), float(lo)] for la, lo in zip(lat_c, lon_c)]))

    bands = []
    fill_steps = max(1, int(fill_steps))
    if zmax > zmin:
        fill_levels = np.linspace(zmin, zmax, fill_steps + 1)
        csf = ax.contourf(GX, GY, Zm, levels=fill_levels)
        for i, segs in enumerate(csf.allsegs):
            rings = []
            for seg in segs:
                if len(seg) < 3:
                    continue
                lon_c, lat_c = inv.transform(seg[:, 0], seg[:, 1])
                rings.append([[float(la), float(lo)] for la, lo in zip(lat_c, lon_c)])
            if rings:
                lo_level = float(fill_levels[i])
                hi_level = float(fill_levels[i + 1]) if i + 1 < len(fill_levels) else zmax
                bands.append(dict(lo=lo_level, hi=hi_level, rings=rings))
    plt.close(fig)
    return dict(lines=lines, bands=bands, zmin=zmin, zmax=zmax)


def read_waterline_points(path):
    """Читает точки уреза воды из файла: CSV с колонками lat/lon (любой разделитель,
    как у GNSS-CSV в sl2sync.read_csv_track), либо обычный текст — по два числа
    (lat, lon) в строке, разделённых запятой/точкой с запятой/пробелом/табом.
    Строки, которые не удаётся разобрать (заголовок, пустые строки), пропускаются."""
    with open(path, newline="", errors="ignore") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            has_header = csv.Sniffer().has_header(sample)
            rows = list(csv.reader(fh, dialect=dialect))
        except csv.Error:
            fh.seek(0)
            has_header = False
            rows = [re.split(r"[,;\s]+", line.strip()) for line in fh if line.strip()]

    lat_idx, lon_idx = 0, 1
    if rows and has_header:
        header = [c.strip().lower() for c in rows[0]]
        lat_idx = next((i for i, c in enumerate(header) if "lat" in c), 0)
        lon_idx = next((i for i, c in enumerate(header) if "lon" in c or "lng" in c), 1)
        rows = rows[1:]

    points = []
    for row in rows:
        if len(row) <= max(lat_idx, lon_idx):
            continue
        try:
            lat = float(row[lat_idx].strip())
            lon = float(row[lon_idx].strip())
        except (ValueError, IndexError):
            continue
        points.append((lat, lon))
    if not points:
        raise ValueError("Не удалось прочитать точки уреза из файла")
    return points


def build_isobaths_map_html(basemap, lat, lon):
    tile = BASEMAPS[basemap]
    center = [sum(lat) / len(lat), sum(lon) / len(lon)] if lat else [0, 0]
    pts = list(zip(lat, lon))
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<style>html,body,#map{{height:100%;margin:0}}
.iso-line-label{{background:rgba(255,255,255,.85);color:#000;padding:0 3px;
               border-radius:2px;white-space:nowrap;font-family:sans-serif;line-height:1.4}}
.iso-area-label{{background:rgba(0,0,0,.55);color:#fff;padding:1px 5px;
               border-radius:3px;white-space:nowrap;font-family:sans-serif;line-height:1.4}}
.iso-area-label-plain{{color:#fff;white-space:nowrap;font-family:sans-serif;line-height:1.4;
               text-shadow:-1px -1px 2px #000,1px -1px 2px #000,-1px 1px 2px #000,1px 1px 2px #000}}
.depth-legend{{background:rgba(20,20,20,.65);color:#fff;padding:8px;border-radius:4px;
               font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.4);
               display:flex;flex-direction:column;align-items:center}}
.depth-legend .bar-wrap{{position:relative;width:12px;height:120px}}
.depth-legend .bar{{width:12px;height:120px}}
.depth-legend .tick{{position:absolute;left:0;width:12px;height:1px;
               background:rgba(0,0,0,.55)}}
.depth-legend .scale{{display:flex;flex-direction:column;justify-content:space-between;
               height:120px;text-align:center}}
.depth-legend .row{{display:flex;align-items:stretch}}</style>
</head><body>
<div id="map"></div>
<script>
var map = L.map('map', {{preferCanvas: true, attributionControl: false}}).setView([{center[0]}, {center[1]}], 15);
L.tileLayer('{tile["url"]}', {{maxZoom: {tile["max_zoom"]}}}).addTo(map);
var pts = {json.dumps(pts)};
var ptsLayer = L.featureGroup();
pts.forEach(function (p) {{
  L.circleMarker([p[0], p[1]], {{radius: 2, weight: 0, fillColor: '#ff0000', fillOpacity: 0.7}}).addTo(ptsLayer);
}});
ptsLayer.addTo(map);
if (pts.length) {{ map.fitBounds(ptsLayer.getBounds()); }}

function setTrackVisible(v) {{
  if (v) {{ if (!map.hasLayer(ptsLayer)) {{ map.addLayer(ptsLayer); }} }}
  else {{ if (map.hasLayer(ptsLayer)) {{ map.removeLayer(ptsLayer); }} }}
}}

var drawMode = false;
var waterline = [];
var waterlineLine = L.polyline([], {{color: '#ffa500', weight: 3}}).addTo(map);
var waterlineMarkers = L.layerGroup().addTo(map);

function setDrawMode(v) {{ drawMode = v; }}
function clearWaterline() {{
  waterline = [];
  waterlineLine.setLatLngs([]);
  waterlineMarkers.clearLayers();
}}
function setWaterline(points) {{
  waterline = points.slice();
  waterlineLine.setLatLngs(waterline);
  waterlineMarkers.clearLayers();
  waterline.forEach(function (p) {{
    L.circleMarker(p, {{radius: 3, color: '#ffa500', fillColor: '#ffa500', fillOpacity: 1}}).addTo(waterlineMarkers);
  }});
}}

var bridge = null;
new QWebChannel(qt.webChannelTransport, function (channel) {{ bridge = channel.objects.bridge; }});

map.on('click', function (e) {{
  if (!drawMode) {{ return; }}
  waterline.push([e.latlng.lat, e.latlng.lng]);
  waterlineLine.setLatLngs(waterline);
  L.circleMarker(e.latlng, {{radius: 3, color: '#ffa500', fillColor: '#ffa500', fillOpacity: 1}}).addTo(waterlineMarkers);
  if (bridge) {{ bridge.mapClicked(e.latlng.lat, e.latlng.lng); }}
}});

var fillLayer = L.layerGroup().addTo(map);
var linesLayer = L.layerGroup().addTo(map);
var labelsLayer = L.layerGroup().addTo(map);

function clearIsobaths() {{
  fillLayer.clearLayers();
  linesLayer.clearLayers();
  labelsLayer.clearLayers();
}}

function setLayerVisible(layer, visible) {{
  if (visible) {{ if (!map.hasLayer(layer)) {{ map.addLayer(layer); }} }}
  else {{ if (map.hasLayer(layer)) {{ map.removeLayer(layer); }} }}
}}
function setLinesVisible(v) {{ setLayerVisible(linesLayer, v); }}
function setFillVisible(v) {{ setLayerVisible(fillLayer, v); }}
function setLabelsVisible(v) {{ setLayerVisible(labelsLayer, v); }}

var fillLegend = L.control({{position: 'bottomright'}});
fillLegend.onAdd = function () {{
  var div = L.DomUtil.create('div', 'depth-legend');
  div.id = 'fill-legend-content';
  return div;
}};
var fillLegendAdded = false;
var lastFillZmin = null, lastFillZmax = null, lastFillColors = [];

function renderFillLegend(zmin, zmax, colors) {{
  var div = document.getElementById('fill-legend-content');
  if (!div) {{ return; }}
  var nTicks = 5;
  // Градиент строим из реальных цветов диапазонов заливки (а не заново по
  // hue 0→220) — иначе 2-стопный CSS-градиент интерполируется в RGB и не
  // совпадает с фактической последовательностью цветов на карте.
  var stops = (colors && colors.length) ? colors.join(', ') : 'hsl(0,75%,50%), hsl(220,75%,50%)';
  var barHtml = '<div class="bar-wrap"><div class="bar" style="background:' +
    'linear-gradient(to top, ' + stops + ')"></div>';
  var scaleHtml = '<div class="scale">';
  for (var k = 0; k <= nTicks; k++) {{
    barHtml += '<div class="tick" style="top:' + (k / nTicks * 100) + '%"></div>';
    scaleHtml += '<span>' + (zmax - (zmax - zmin) * (k / nTicks)).toFixed(1) + '</span>';
  }}
  barHtml += '</div>';
  scaleHtml += '</div>';
  div.innerHTML = '<div>Глубина, м</div><div class="row">' + barHtml + scaleHtml + '</div>';
}}

function setFillLegendVisible(v) {{
  if (v) {{
    // onAdd создаёт пустой div — если zmin/zmax уже известны (построение могло
    // случиться раньше или позже этого вызова), сразу заполняем содержимое,
    // иначе легенда показывается пустой коробкой.
    if (!fillLegendAdded) {{ fillLegend.addTo(map); fillLegendAdded = true; }}
    if (lastFillZmin != null && lastFillZmax != null) {{
      renderFillLegend(lastFillZmin, lastFillZmax, lastFillColors);
    }}
  }}
  else {{ if (fillLegendAdded) {{ fillLegend.remove(); fillLegendAdded = false; }} }}
}}

function ringCentroid(rings) {{
  var cx = 0, cy = 0, n = 0;
  rings.forEach(function (ring) {{
    ring.forEach(function (p) {{ cx += p[0]; cy += p[1]; n++; }});
  }});
  return n ? [cx / n, cy / n] : null;
}}

function drawIsobaths(payload) {{
  clearIsobaths();
  var style = payload.style || {{}};
  var lineColor = style.lineColor || '#3388ff';
  var lineWidth = style.lineWidth || 2;
  var labelSize = style.labelSize || 11;
  var labelFreq = Math.max(1, style.labelFreq || 40);
  var fillOpacity = style.fillOpacity != null ? style.fillOpacity : 0.5;
  var areaLabelClass = style.labelNoBg ? 'iso-area-label-plain' : 'iso-area-label';
  var lineLabelClass = style.labelNoBg ? 'iso-area-label-plain' : 'iso-line-label';

  // Не повторяем одну и ту же подпись (по тексту), если рядом — в радиусе
  // 4 высот шрифта на экране — уже стоит подпись с тем же значением: на
  // извилистых линиях/соседних диапазонах одно и то же число иначе может
  // напечататься по нескольку раз почти вплотную.
  var placedLabels = [];
  function shouldPlaceLabel(text, latlng) {{
    var pt = map.latLngToLayerPoint(latlng);
    var minDist = 4 * labelSize;
    for (var pi = 0; pi < placedLabels.length; pi++) {{
      if (placedLabels[pi].text === text && pt.distanceTo(placedLabels[pi].pt) < minDist) {{
        return false;
      }}
    }}
    placedLabels.push({{text: text, pt: pt}});
    return true;
  }}

  (payload.bands || []).forEach(function (b) {{
    var validRings = (b.rings || []).filter(function (ring) {{ return ring.length >= 3; }});
    if (validRings.length) {{
      // Все кольца диапазона — одной фигурой с правилом заливки evenodd (по умолчанию у
      // Leaflet Canvas): внешние границы и вложенные «дырки» (где начинается более
      // глубокий диапазон) корректно вычитаются, а не закрашиваются второй раз поверх —
      // иначе прозрачность в местах наложения выглядела заметно плотнее, чем на краю.
      L.polygon(validRings, {{stroke: false, fillColor: b.color, fillOpacity: fillOpacity}}).addTo(fillLayer);
    }}
    var c = ringCentroid(validRings);
    if (c) {{
      var text = b.lo.toFixed(1) + '–' + b.hi.toFixed(1) + ' м';
      if (shouldPlaceLabel(text, c)) {{
        L.marker(c, {{
          icon: L.divIcon({{className: areaLabelClass, iconSize: null,
                            html: '<span style="font-size:' + labelSize + 'px">' + text + '</span>'}}),
          interactive: false
        }}).addTo(labelsLayer);
      }}
    }}
  }});

  (payload.lines || []).forEach(function (c) {{
    // Изобаты на целых метрах — чёрным и непрозрачно (как «жирные» опорные
    // линии на морских картах), промежуточные (дробный шаг изобат) — обычным
    // выбранным цветом, но на 50% прозрачнее, чтобы не спорили с целыми.
    var isWhole = Math.abs(c.level - Math.round(c.level)) < 0.01;
    var segColor = isWhole ? '#000000' : lineColor;
    var segOpacity = isWhole ? 1.0 : 0.5;
    L.polyline(c.coords, {{color: segColor, weight: lineWidth, opacity: segOpacity}}).addTo(linesLayer);
    for (var i = 0; i < c.coords.length; i += labelFreq) {{
      var text = c.level.toFixed(1);
      if (shouldPlaceLabel(text, c.coords[i])) {{
        L.marker(c.coords[i], {{
          icon: L.divIcon({{className: lineLabelClass, iconSize: null,
                            html: '<span style="font-size:' + labelSize + 'px">' + text + '</span>'}}),
          interactive: false
        }}).addTo(labelsLayer);
      }}
    }}
  }});

  if (payload.zmin != null && payload.zmax != null) {{
    lastFillZmin = payload.zmin;
    lastFillZmax = payload.zmax;
    lastFillColors = (payload.bands || []).map(function (b) {{ return b.color; }});
    renderFillLegend(payload.zmin, payload.zmax, lastFillColors);
  }}
}}
</script>
</body></html>"""


def build_isobaths_js(result, style):
    payload = dict(lines=result.get("lines", []), style=style,
                    zmin=result.get("zmin"), zmax=result.get("zmax"))
    bands = result.get("bands", [])
    n = len(bands)
    payload["bands"] = []
    for i, b in enumerate(bands):
        t = i / max(1, n - 1)
        color = f"hsl({int(220 * t)},75%,50%)"
        payload["bands"].append(dict(lo=b["lo"], hi=b["hi"], rings=b["rings"], color=color))
    return f"drawIsobaths({json.dumps(payload)});"


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
    # t_abs — реальное UTC-время (эпоха GNSS-трека), в отличие от mtime файла
    # .sl2, которое зависит от часов эхолота/карты памяти и может быть неверным
    # (см. docs/CONTEXT.md) — раз GNSS-трек всё равно уже загружен и
    # синхронизирован, отдаём и уточнённые начало/конец записи.
    start_utc = datetime.utcfromtimestamp(float(t_abs.min()))
    end_utc = datetime.utcfromtimestamp(float(t_abs.max()))
    span_s = float(s["t_rel"][-1] - s["t_rel"][0])

    depth_before = s["depth_m"][has_gps]
    depth_after = s["depth_m"][ok]
    depth_valid = s["depth_m"][np.isfinite(s["depth_m"]) & (s["depth_m"] > 0)]
    depth_vmin = float(np.nanmin(depth_valid)) if len(depth_valid) else 0.0
    depth_vmax = float(np.nanmax(depth_valid)) if len(depth_valid) else 1.0
    # has_gps и ok — разные маски по одному и тому же массиву пингов (has_gps —
    # был ли валиден встроенный GPS Lowrance у пинга, ok — нашлась ли для пинга
    # интерполированная позиция на GNSS-треке после синхронизации), поэтому
    # точка i массива "before" и точка i массива "after" — не обязательно один
    # и тот же пинг. Отдаём настоящий номер пинга (позицию в полном массиве)
    # для каждой точки — по нему в GUI ищется ближайшее совпадение при
    # подсветке одной и той же точки на обеих картах сравнения.
    ping_idx_before = np.flatnonzero(has_gps)
    ping_idx_after = np.flatnonzero(ok)

    return dict(
        offset_s=tm["a"], rms_pos=tm.get("rms_pos"),
        # Подробности расчёта — для детального отчёта в OffsetDialog: грубое
        # смещение и качество кросс-корреляции скорости (NCC), дрейф часов
        # эхолота (ppm и секунды за запись, по скольким окнам с поворотами
        # оценён — 0, если окон не хватило, см. estimate_time_model), и была
        # ли уточнённая точка на краю окна поиска (edge=True — возможно,
        # ненадёжно, стоит проверить).
        coarse_offset_s=tm.get("coarse"), ncc=tm.get("ncc"),
        drift_ppm=tm.get("b", 0.0) * 1e6, drift_s=tm.get("b", 0.0) * span_s,
        drift_windows=tm.get("drift_windows", 0), edge=tm.get("edge"),
        start=start_utc.isoformat(), end=end_utc.isoformat(),
        duration_s=(end_utc - start_utc).total_seconds(),
        depth_vmin=depth_vmin, depth_vmax=depth_vmax,
        before=dict(
            sl2=dict(lat=s["lat"][has_gps].tolist(), lon=s["lon"][has_gps].tolist(),
                     depth=depth_before.tolist(), ping_idx=ping_idx_before.tolist()),
            gnss=dict(lat=g["lat"].tolist(), lon=g["lon"].tolist())),
        after=dict(
            sl2=dict(lat=lat2.tolist(), lon=lon2.tolist(), depth=depth_after.tolist(),
                     ping_idx=ping_idx_after.tolist()),
            gnss=dict(lat=g["lat"].tolist(), lon=g["lon"].tolist())),
    )


class DepthRulerWidget(QWidget):
    """Линейка слева от эхограммы — отдельный виджет вне области горизонтальной
    прокрутки, чтобы не уезжать вместе с картинкой. По вертикали синхронизируется
    со скроллом канваса (see EchogramViewDialog).

    Показывает глубину в метрах, если эхограмма откалибрована по полю диапазона
    сонара (range_m, см. build_echogram_image) — так почти всегда, кроме случая,
    когда это поле не удалось прочитать; тогда, как раньше, показывается номер
    байта от начала пинга без домысленной привязки к метрам."""
    WIDTH = 55

    def __init__(self, canvas):
        super().__init__()
        self.canvas = canvas
        self.setFixedWidth(self.WIDTH)
        self.row_count = 0
        self.content_height = 0
        self.range_m = None
        self.scroll_y = 0

    def set_params(self, row_count, content_height, range_m=None):
        self.row_count = row_count
        self.content_height = content_height
        self.range_m = range_m
        self.update()

    def set_scroll_offset(self, y):
        self.scroll_y = y
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("black"))
        if self.content_height <= 0:
            return
        painter.setPen(QColor("#ccc"))
        h = self.height()
        ticks = max(2, min(12, h // 40))
        for k in range(ticks + 1):
            y_viewport = k / ticks * h
            y_content = self.scroll_y + y_viewport
            if y_content > self.content_height:
                continue
            row = (y_content / self.content_height) * self.row_count
            painter.drawLine(self.WIDTH - 5, int(y_viewport), self.WIDTH, int(y_viewport))
            if self.range_m is not None:
                label = f"{row / self.row_count * self.range_m:.1f}"
            else:
                label = f"{row:.0f}"
            painter.drawText(2, min(int(y_viewport) + 4, h - 2), label)

    def wheelEvent(self, event):
        self.canvas.wheelEvent(event)


class EchoTimelineWidget(QWidget):
    """Профиль глубины дна (по данным сонара, depth_m — единственное здесь
    откалиброванное значение) под эхограммой. По горизонтали синхронизирован
    со скроллом канваса, как линейка — по вертикали."""
    HEIGHT = 60

    def __init__(self):
        super().__init__()
        self.setFixedHeight(self.HEIGHT)
        self.depth_m = []
        self.content_width = 0
        self.scroll_x = 0
        self.hover_col = None

    def set_params(self, depth_m, content_width):
        self.depth_m = depth_m
        self.content_width = content_width
        self.update()

    def set_scroll_offset(self, x):
        self.scroll_x = x
        self.update()

    def set_hover_col(self, col):
        self.hover_col = col
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#151515"))
        n = len(self.depth_m)
        if n < 2 or self.content_width <= 0:
            return
        w, h = self.width(), self.height()
        dmin, dmax = min(self.depth_m), max(self.depth_m)
        if dmax <= dmin:
            dmax = dmin + 1e-6
        per_ping = self.content_width / n
        i0 = max(0, int(self.scroll_x / per_ping) - 1)
        i1 = min(n, int((self.scroll_x + w) / per_ping) + 2)
        painter.setPen(QColor("#3388ff"))
        prev = None
        for i in range(i0, i1):
            x = i * per_ping - self.scroll_x
            t = (self.depth_m[i] - dmin) / (dmax - dmin)
            y = 4 + t * (h - 8)
            if prev is not None:
                painter.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
            prev = (x, y)
        if self.hover_col is not None and 0 <= self.hover_col < n:
            x = self.hover_col * per_ping - self.scroll_x
            painter.setPen(QColor(255, 255, 0, 200))
            painter.drawLine(int(x), 0, int(x), h)
            painter.setPen(QColor("white"))
            painter.drawText(min(w - 60, max(2, int(x) + 4)), 13,
                              f"{self.depth_m[self.hover_col]:.2f} м")


class EchogramCanvas(QWidget):
    AXIS_H = 22
    zoom_changed = Signal()
    hover_changed = Signal(object)  # индекс пинга под курсором, либо None

    def __init__(self):
        super().__init__()
        self.arr = None
        self.valid = None
        self.image = None
        self.depth_m = []
        self.range_m = None
        self.axis_vals = []
        self.axis_mode = "time"
        self.zoom = 1.0
        self.contrast = 1.0
        self.hover_pos = None
        self.setMouseTracking(True)

    def set_data(self, arr, valid, depth_m, axis_vals, axis_mode, range_m=None):
        self.arr = arr
        self.valid = valid
        self.depth_m = depth_m
        self.range_m = range_m
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
        # При уменьшении масштаба сначала огрубляем усреднением по блокам (см.
        # block_reduce_mean) — иначе на отрисовке эхограммы образуется алиасинг
        # (зубчатые вертикальные полосы) вместо гладкой картины, которая видна
        # при масштабе 1:1. При zoom >= 1 показываем как есть, пиксель в пиксель
        # (или крупнее — тут дробление не нужно).
        factor = max(1, round(1.0 / self.zoom)) if self.zoom < 1 else 1
        arr, valid = block_reduce_mean(self.arr, self.valid, factor, factor)
        adj = np.clip((arr.astype(np.float32) - 128.0) * self.contrast + 128.0, 0, 255)
        self.image = colorize_echogram(adj.astype(np.uint8), valid)

    def sizeHint(self):
        if self.arr is None:
            return QSize(400, 200)
        rows, cols = self.arr.shape
        return QSize(int(cols * self.zoom), int(rows * self.zoom) + self.AXIS_H)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("black"))
        if self.image is None or self.arr is None:
            painter.setPen(QColor("white"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Чтение файла…")
            return
        rows, cols = self.arr.shape
        w = int(cols * self.zoom)
        h = int(rows * self.zoom)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(QRect(0, 0, w, h), self.image)
        self._draw_axis(painter, w, h)
        if self.hover_pos is not None:
            self._draw_crosshair(painter, w, h)

    def _draw_crosshair(self, painter, w, h):
        x, y = self.hover_pos
        if not (0 <= x <= w and 0 <= y <= h):
            return
        painter.setPen(QColor(255, 255, 0, 200))
        painter.drawLine(0, int(y), w, int(y))
        painter.drawLine(int(x), 0, int(x), h)
        row = y / h * self.arr.shape[0] if h else 0
        col = int(x / w * len(self.axis_vals)) if w and self.axis_vals else 0
        col = max(0, min(len(self.axis_vals) - 1, col)) if self.axis_vals else 0
        if self.range_m is not None:
            row_label = f"{row / self.arr.shape[0] * self.range_m:.1f} м"
            row_box_w = 44
        else:
            row_label = f"{row:.0f}"
            row_box_w = 34
        col_label = format_axis_label(self.axis_vals[col], self.axis_mode) if self.axis_vals else ""
        painter.setBrush(QColor(255, 255, 0, 220))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(2, max(0, int(y) - 14), row_box_w, 14)
        painter.drawRect(min(w - 46, int(x) + 2), 2, 44, 14)
        painter.setPen(QColor("black"))
        painter.drawText(4, max(11, int(y) - 3), row_label)
        painter.drawText(min(w - 44, int(x) + 4), 13, col_label)

    def mouseMoveEvent(self, event):
        if self.image is None or self.arr is None:
            return
        pos = event.position() if hasattr(event, "position") else event.pos()
        self.hover_pos = (pos.x(), pos.y())
        w = int(self.arr.shape[1] * self.zoom)
        col = int(pos.x() / w * len(self.axis_vals)) if w and self.axis_vals else None
        if col is not None:
            col = max(0, min(len(self.axis_vals) - 1, col))
        self.hover_changed.emit(col)
        self.update()

    def leaveEvent(self, event):
        self.hover_pos = None
        self.hover_changed.emit(None)
        self.update()

    def _draw_axis(self, painter, w, h):
        painter.setPen(QColor("#ccc"))
        if len(self.axis_vals) < 2:
            return
        a0, a1 = self.axis_vals[0], self.axis_vals[-1]
        ticks = max(2, min(10, w // 90))
        for k in range(ticks + 1):
            frac = k / ticks
            x = frac * w
            label = format_axis_label(a0 + frac * (a1 - a0), self.axis_mode)
            painter.drawLine(int(x), h, int(x), h + 4)
            painter.drawText(int(x) - 15, h + self.AXIS_H - 4, label)

    def wheelEvent(self, event):
        if self.image is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom = max(0.1, min(20.0, self.zoom * factor))
        self._rebuild_image()
        self.updateGeometry()
        self.resize(self.sizeHint())
        self.update()
        self.zoom_changed.emit()
        event.accept()


class EchogramViewDialog(QDialog):
    def __init__(self, parent, sl2_path, axis_mode="time"):
        super().__init__(parent)
        self.setWindowTitle(f"Эхограмма — {os.path.basename(sl2_path)}")
        self.resize(1000, 500)
        self.axis_mode = axis_mode

        self.status_label = QLabel("Чтение файла…")
        self.canvas = EchogramCanvas()
        self.canvas.zoom_changed.connect(self.sync_side_widgets)
        self.ruler = DepthRulerWidget(self.canvas)
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        scroll.verticalScrollBar().valueChanged.connect(self.ruler.set_scroll_offset)

        content_row = QHBoxLayout()
        content_row.addWidget(self.ruler)
        content_row.addWidget(scroll, 1)

        self.echo_timeline = EchoTimelineWidget()
        self.canvas.hover_changed.connect(self.echo_timeline.set_hover_col)
        scroll.horizontalScrollBar().valueChanged.connect(self.echo_timeline.set_scroll_offset)
        timeline_row = QHBoxLayout()
        timeline_row.addSpacing(DepthRulerWidget.WIDTH)
        timeline_row.addWidget(self.echo_timeline, 1)

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
        layout.addLayout(content_row, 1)
        layout.addLayout(timeline_row)
        layout.addLayout(contrast_row)
        self.setLayout(layout)

        run_async(lambda: read_echogram_waterfall(sl2_path),
                  on_finished=self.on_loaded, on_error=self.on_error)

    def sync_side_widgets(self):
        if self.canvas.image is None or self.canvas.arr is None:
            return
        rows, cols = self.canvas.arr.shape
        row_count = rows
        content_height = int(rows * self.canvas.zoom)
        self.ruler.set_params(row_count, content_height, self.canvas.range_m)
        content_width = int(cols * self.canvas.zoom)
        self.echo_timeline.set_params(self.canvas.depth_m, content_width)

    def on_loaded(self, data):
        arr, valid, cal_range = build_echogram_image(data["records"], data["range_m"])
        if arr is None:
            self.status_label.setText("В файле нет данных эхограммы")
            return
        note = ""
        if data["mixed_freqs"] > 1:
            note = (f" — в канале {data['mixed_freqs']} частоты, показана самая частая, "
                    f"остальные пинги пропущены")
        if cal_range is not None:
            note += (f"; глубина откалибрована по диапазону сонара "
                      f"(0–{cal_range:.1f} м)")
        self.status_label.setText(f"{arr.shape[1]} пингов, {arr.shape[0]} байт по глубине "
                                   f"(колесо мыши — масштаб){note}")
        axis_vals = data["dist"] if self.axis_mode == "distance" else data["t_rel"]
        self.canvas.set_data(arr, valid, data["depth_m"], axis_vals, self.axis_mode, cal_range)
        self.sync_side_widgets()

    def on_error(self, message):
        self.status_label.setText(f"Ошибка чтения: {message}")


class MapClickBridge(QObject):
    """Мост QWebChannel: клик по карте в JS вызывает mapClicked прямо в
    Python (тот же поток, без QThread — сюда правило про сигналы между
    потоками и QWebEngineView не относится)."""

    def __init__(self, on_click):
        super().__init__()
        self.on_click = on_click

    @Slot(float, float)
    def mapClicked(self, lat, lon):
        self.on_click(lat, lon)


class TrackCompareBridge(QObject):
    """Мост QWebChannel между двумя картами сравнения треков (OffsetDialog,
    build_tracks_html с sync=True): движение и наведение на точку на одной
    карте транслируется на вторую через runJavaScript — сами карты это две
    разные страницы в разных QWebEngineView, без прямой связи между JS."""

    def __init__(self, other_view_getter):
        super().__init__()
        self._other = other_view_getter

    @Slot(float, float, float)
    def viewChanged(self, lat, lng, zoom):
        self._other().page().runJavaScript(f"syncView({lat}, {lng}, {zoom});")

    @Slot(int)
    def hoverPoint(self, idx):
        self._other().page().runJavaScript(f"remoteHighlight({idx});")

    @Slot()
    def hoverEnd(self):
        self._other().page().runJavaScript("remoteHighlight(-1);")


class IsobathsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Построение изобат")
        self.resize(1150, 700)
        self.main_window = parent
        self.waterline_points = []
        self.last_result = None
        self.line_color = QColor("#000000")

        self.bridge = MapClickBridge(self.on_map_clicked)
        self.channel = QWebChannel()
        self.channel.registerObject("bridge", self.bridge)
        self.map_view = new_map_view()
        self.map_view.page().setWebChannel(self.channel)

        points = self.main_window.points or {}
        html = build_isobaths_map_html(self.main_window.basemap_combo.currentText(),
                                        points.get("lat", []), points.get("lon", []))
        load_html(self.map_view, html, "isobaths")

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 100.0)
        self.interval_spin.setValue(1.0)
        self.interval_spin.setSuffix(" м")
        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("Шаг изобат:"))
        interval_row.addWidget(self.interval_spin)

        self.cell_spin = QDoubleSpinBox()
        self.cell_spin.setRange(0.1, 1000.0)
        self.cell_spin.setValue(1.0)
        self.cell_spin.setSuffix(" м")
        cell_row = QHBoxLayout()
        cell_row.addWidget(QLabel("Ячейка сетки:"))
        cell_row.addWidget(self.cell_spin)

        self.smooth_spin = QDoubleSpinBox()
        self.smooth_spin.setRange(0.0, 20.0)
        self.smooth_spin.setSingleStep(0.5)
        self.smooth_spin.setValue(0.0)
        self.smooth_spin.setSuffix(" яч.")
        self.smooth_spin.setToolTip("Сглаживание грида (гауссово размытие в ячейках сетки) "
                                     "перед построением линий и заливки — 0 значит без сглаживания. "
                                     "Действует после нажатия «Построить».")
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Сглаживание:"))
        smooth_row.addWidget(self.smooth_spin)

        self.hide_track_btn = QPushButton("Скрыть трек")
        self.hide_track_btn.setCheckable(True)
        self.hide_track_btn.toggled.connect(self.on_hide_track_toggled)

        load_waterline_btn = QPushButton("Загрузить урез из файла")
        load_waterline_btn.clicked.connect(self.load_waterline_from_file)
        load_waterline_row = QHBoxLayout()
        load_waterline_row.addWidget(load_waterline_btn, 1)
        load_waterline_row.addWidget(help_icon(WATERLINE_FILE_HELP))

        self.draw_btn = QPushButton("Нарисовать урез воды")
        self.draw_btn.setCheckable(True)
        self.draw_btn.toggled.connect(self.on_draw_toggled)

        clear_waterline_btn = QPushButton("Очистить урез")
        clear_waterline_btn.clicked.connect(self.clear_waterline)

        self.waterline_label = QLabel("Точек уреза: 0")

        line_group = QGroupBox("Линии изобат")
        self.show_lines_check = QCheckBox("Отображать линии")
        self.show_lines_check.setChecked(True)
        self.show_lines_check.toggled.connect(self.on_show_lines_toggled)
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.25, 10.0)
        self.line_width_spin.setSingleStep(0.25)
        self.line_width_spin.setDecimals(2)
        self.line_width_spin.setValue(1.0)
        self.line_width_spin.valueChanged.connect(self.apply_style)
        line_width_row = QHBoxLayout()
        line_width_row.addWidget(QLabel("Толщина:"))
        line_width_row.addWidget(self.line_width_spin)
        self.line_color_btn = QPushButton()
        self.line_color_btn.clicked.connect(self.pick_line_color)
        self._update_line_color_btn()
        line_color_row = QHBoxLayout()
        line_color_row.addWidget(QLabel("Цвет:"))
        line_color_row.addWidget(self.line_color_btn)
        line_layout = QVBoxLayout()
        line_layout.addWidget(self.show_lines_check)
        line_layout.addLayout(line_width_row)
        line_layout.addLayout(line_color_row)
        line_group.setLayout(line_layout)

        label_group = QGroupBox("Подписи изобат")
        self.show_labels_check = QCheckBox("Отображать подписи")
        self.show_labels_check.setChecked(True)
        self.show_labels_check.toggled.connect(self.on_show_labels_toggled)
        self.label_size_spin = QSpinBox()
        self.label_size_spin.setRange(6, 30)
        self.label_size_spin.setValue(11)
        self.label_size_spin.valueChanged.connect(self.apply_style)
        label_size_row = QHBoxLayout()
        label_size_row.addWidget(QLabel("Размер шрифта:"))
        label_size_row.addWidget(self.label_size_spin)
        self.label_freq_spin = QSpinBox()
        self.label_freq_spin.setRange(5, 500)
        self.label_freq_spin.setValue(40)
        self.label_freq_spin.valueChanged.connect(self.apply_style)
        label_freq_row = QHBoxLayout()
        label_freq_row.addWidget(QLabel("Частота (точек):"))
        label_freq_row.addWidget(self.label_freq_spin)
        self.label_nobg_check = QCheckBox("Только цифры, без фона")
        self.label_nobg_check.setChecked(False)
        self.label_nobg_check.toggled.connect(self.apply_style)
        label_layout = QVBoxLayout()
        label_layout.addWidget(self.show_labels_check)
        label_layout.addLayout(label_size_row)
        label_layout.addLayout(label_freq_row)
        label_layout.addWidget(QLabel("Подписи диапазонов глубин внутри заливки —\n"
                                       "тем же шрифтом, в центре каждой области."))
        label_layout.addWidget(self.label_nobg_check)
        label_group.setLayout(label_layout)

        fill_group = QGroupBox("Заливка изобат")
        self.show_fill_check = QCheckBox("Отображать заливку")
        self.show_fill_check.setChecked(True)
        self.show_fill_check.toggled.connect(self.on_show_fill_toggled)
        self.fill_opacity_spin = QSpinBox()
        self.fill_opacity_spin.setRange(0, 100)
        self.fill_opacity_spin.setValue(50)
        self.fill_opacity_spin.setSuffix(" %")
        self.fill_opacity_spin.valueChanged.connect(self.apply_style)
        fill_opacity_row = QHBoxLayout()
        fill_opacity_row.addWidget(QLabel("Прозрачность:"))
        fill_opacity_row.addWidget(self.fill_opacity_spin)
        self.fill_steps_spin = QSpinBox()
        self.fill_steps_spin.setRange(2, 30)
        self.fill_steps_spin.setValue(8)
        fill_steps_row = QHBoxLayout()
        fill_steps_row.addWidget(QLabel("Ступеней:"))
        fill_steps_row.addWidget(self.fill_steps_spin)
        fill_layout = QVBoxLayout()
        fill_layout.addWidget(self.show_fill_check)
        fill_layout.addLayout(fill_opacity_row)
        fill_layout.addLayout(fill_steps_row)
        fill_group.setLayout(fill_layout)

        settings_layout = QVBoxLayout()
        settings_layout.addWidget(self.hide_track_btn)
        settings_layout.addLayout(interval_row)
        settings_layout.addLayout(cell_row)
        settings_layout.addLayout(smooth_row)
        settings_layout.addLayout(load_waterline_row)
        settings_layout.addWidget(self.draw_btn)
        settings_layout.addWidget(clear_waterline_btn)
        settings_layout.addWidget(self.waterline_label)
        settings_layout.addWidget(line_group)
        settings_layout.addWidget(label_group)
        settings_layout.addWidget(fill_group)
        settings_layout.addStretch(1)
        settings_widget = QWidget()
        settings_widget.setLayout(settings_layout)
        settings_widget.setFixedWidth(240)

        content_row = QHBoxLayout()
        content_row.addWidget(settings_widget)
        content_row.addWidget(self.map_view, 1)

        self.status_label = QLabel("")

        build_btn = QPushButton("Построить")
        build_btn.clicked.connect(self.build_isobaths)
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setFrameShadow(QFrame.Shadow.Sunken)
        delete_btn = QPushButton("Удалить изобаты")
        delete_btn.clicked.connect(self.delete_isobaths)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)

        save_btn = QPushButton("Сохранить")
        save_menu = QMenu(save_btn)
        save_menu.addAction("Изображение (PNG)…", self.save_as_image)
        save_btn.setMenu(save_menu)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(build_btn)
        bottom_row.addWidget(sep1)
        bottom_row.addWidget(delete_btn)
        bottom_row.addWidget(sep2)
        bottom_row.addWidget(save_btn)
        bottom_row.addStretch(1)

        layout = QVBoxLayout()
        layout.addLayout(content_row, 1)
        layout.addWidget(self.status_label)
        layout.addLayout(bottom_row)
        self.setLayout(layout)

    def on_draw_toggled(self, checked):
        self.draw_btn.setText("Рисование уреза: кликните по карте (ещё раз — выкл)"
                               if checked else "Нарисовать урез воды")
        self.map_view.page().runJavaScript(f"setDrawMode({'true' if checked else 'false'});")

    def on_map_clicked(self, lat, lon):
        if not self.draw_btn.isChecked():
            return
        self.waterline_points.append((lat, lon))
        self.waterline_label.setText(f"Точек уреза: {len(self.waterline_points)}")

    def clear_waterline(self):
        self.waterline_points = []
        self.waterline_label.setText("Точек уреза: 0")
        self.map_view.page().runJavaScript("clearWaterline();")

    def on_hide_track_toggled(self, checked):
        self.hide_track_btn.setText("Показать трек" if checked else "Скрыть трек")
        self.map_view.page().runJavaScript(f"setTrackVisible({'false' if checked else 'true'});")

    def load_waterline_from_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить урез воды", "",
                                               "Текст/CSV (*.csv *.txt);;Все файлы (*)")
        if not path:
            return
        try:
            points = read_waterline_points(path)
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "Урез воды", f"Не удалось прочитать файл:\n{e}")
            return
        self.waterline_points.extend(points)
        self.waterline_label.setText(f"Точек уреза: {len(self.waterline_points)}")
        points_js = json.dumps([[lat, lon] for lat, lon in self.waterline_points])
        self.map_view.page().runJavaScript(f"setWaterline({points_js});")

    def _update_line_color_btn(self):
        self.line_color_btn.setStyleSheet(f"background-color: {self.line_color.name()};")
        self.line_color_btn.setText(self.line_color.name())

    def pick_line_color(self):
        color = QColorDialog.getColor(self.line_color, self, "Цвет изобат")
        if color.isValid():
            self.line_color = color
            self._update_line_color_btn()
            self.apply_style()

    def current_style(self):
        return dict(lineColor=self.line_color.name(),
                    lineWidth=self.line_width_spin.value(),
                    labelSize=self.label_size_spin.value(),
                    labelFreq=self.label_freq_spin.value(),
                    fillOpacity=self.fill_opacity_spin.value() / 100.0,
                    labelNoBg=self.label_nobg_check.isChecked())

    def apply_style(self, *_args):
        if self.last_result is None:
            return
        self.map_view.page().runJavaScript(build_isobaths_js(self.last_result, self.current_style()))

    def on_show_lines_toggled(self, checked):
        self.map_view.page().runJavaScript(f"setLinesVisible({'true' if checked else 'false'});")

    def on_show_fill_toggled(self, checked):
        js = "true" if checked else "false"
        self.map_view.page().runJavaScript(f"setFillVisible({js}); setFillLegendVisible({js});")

    def on_show_labels_toggled(self, checked):
        self.map_view.page().runJavaScript(f"setLabelsVisible({'true' if checked else 'false'});")

    def build_isobaths(self):
        points = self.main_window.points or {}
        lat, lon, depth = points.get("lat", []), points.get("lon", []), points.get("value", [])
        if len(lat) < 10:
            QMessageBox.information(self, "Изобаты",
                                     "Недостаточно точек — сначала загрузите эхограмму.")
            return
        self.status_label.setText("Строю изобаты…")
        run_async(lambda: compute_isobaths(lat, lon, depth, list(self.waterline_points),
                                            self.cell_spin.value(), self.interval_spin.value(),
                                            self.fill_steps_spin.value(), self.smooth_spin.value()),
                  on_finished=self.on_built, on_error=self.on_build_error)

    def on_built(self, result):
        self.last_result = result
        self.status_label.setText(f"Построено изобат: {len(result['lines'])}, "
                                   f"диапазонов заливки: {len(result['bands'])}")
        self.apply_style()
        self.on_show_lines_toggled(self.show_lines_check.isChecked())
        self.on_show_fill_toggled(self.show_fill_check.isChecked())
        self.on_show_labels_toggled(self.show_labels_check.isChecked())

    def delete_isobaths(self):
        self.last_result = None
        self.status_label.setText("Изобаты удалены")
        self.map_view.page().runJavaScript("clearIsobaths(); setFillLegendVisible(false);")

    def save_as_image(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить изображение", "", "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        if self.map_view.grab().save(path, "PNG"):
            self.status_label.setText(f"Сохранено: {path}")
        else:
            QMessageBox.warning(self, "Сохранение", "Не удалось сохранить изображение.")

    def on_build_error(self, message):
        self.status_label.setText(f"Ошибка: {message}")


class OffsetDialog(QDialog):
    def __init__(self, parent, basemap, result):
        super().__init__(parent)
        self.setWindowTitle("Сравнение треков до/после синхронизации")
        self.resize(1400, 850)

        report_box = QGroupBox("Детали расчёта смещения")
        report_form = QFormLayout()
        report_form.addRow(
            label_with_help(
                "Смещение (уточнённое):",
                "Итоговое смещение времени между часами эхолота и GNSS-приёмником, "
                "секунды. Прибавляется к относительному времени пинга sl2, чтобы "
                "получить настоящее UTC-время. Найдено уточнением по траектории "
                "после грубой оценки по кросс-корреляции скорости (см. ниже)."),
            QLabel(f"{result['offset_s']:+.3f} с"))

        coarse, ncc = result.get("coarse_offset_s"), result.get("ncc")
        if coarse is not None and ncc is not None:
            if ncc < 0.5:
                qual = "слабая — проверьте, что записи с одного выхода"
            elif ncc < 0.7:
                qual = "средняя"
            else:
                qual = "хорошая"
            report_form.addRow(
                label_with_help(
                    "Грубое смещение (по скорости):",
                    "Первая, приближённая оценка смещения — по максимуму "
                    "кросс-корреляции профилей скорости GNSS-трека и встроенного "
                    "GPS Lowrance. NCC (normalized cross-correlation) — качество "
                    "совпадения профилей, от 0 до 1: чем выше, тем увереннее "
                    "найдено смещение. От этой точки идёт дальнейшее уточнение."),
                QLabel(f"{coarse:+.2f} с, корреляция {ncc:.2f} ({qual})"))

        if result.get("rms_pos") is not None:
            report_form.addRow(
                label_with_help(
                    "СКО позиций после синхронизации:",
                    "Среднеквадратичное отклонение между положениями встроенного "
                    "GPS Lowrance и GNSS-трека после применения найденного "
                    "смещения — грубая оценка того, насколько хорошо совпали "
                    "треки, в метрах. Это ошибка позиционирования (зависит от "
                    "качества GPS-фикса), а не ошибка глубины."),
                QLabel(f"{result['rms_pos']:.2f} м"))

        if result.get("edge"):
            edge_lbl = QLabel("⚠ уточнённая точка на краю окна поиска — возможно, "
                               "истинное смещение вне проверенного диапазона")
            edge_lbl.setStyleSheet("color: #d9822b;")
            report_form.addRow(
                label_with_help(
                    "",
                    "Уточнение смещения ищет минимум СКО позиций в окне ±1.5 с "
                    "вокруг грубой оценки. Если найденная точка оказалась на "
                    "самом краю этого окна, значит минимум мог быть не пойман "
                    "целиком — стоит перепроверить результат вручную (например, "
                    "через --offset в sl2sync.py с ручным перебором)."),
                edge_lbl)

        drift_windows = result.get("drift_windows", 0)
        drift_help = (
            "Изменение скорости хода часов эхолота относительно GNSS за время "
            "записи (в миллионных долях, ppm) — оценивается по нескольким "
            "окнам записи с поворотами (на прямом галсе смещение по времени "
            "ненаблюдаемо). Это отдельная, обычно небольшая поправка сверх "
            "смещения выше — она накапливается пропорционально времени от "
            "середины записи, а не постоянна."
        )
        if drift_windows >= 3:
            report_form.addRow(
                label_with_help("Дрейф часов эхолота:", drift_help),
                QLabel(f"{result.get('drift_ppm', 0.0):+.1f} ppm "
                       f"({result.get('drift_s', 0.0):+.2f} с за запись), окон: {drift_windows}"))
        else:
            report_form.addRow(
                label_with_help("Дрейф часов эхолота:", drift_help),
                QLabel("не оценивался — мало окон с поворотами "
                       "(нужна запись > 10–15 мин)"))

        if result.get("start") and result.get("end"):
            start_str = datetime.fromisoformat(result["start"]).strftime("%d.%m.%Y %H:%M:%S")
            end_str = datetime.fromisoformat(result["end"]).strftime("%d.%m.%Y %H:%M:%S")
            report_form.addRow(
                label_with_help(
                    "Начало/конец записи (по GNSS):",
                    "Абсолютное время начала и конца записи в UTC, вычисленное "
                    "через найденное смещение и относительное время sl2 — "
                    "точнее, чем время изменения файла на диске (mtime) или "
                    "служебное поле в самом sl2 (см. документацию проекта)."),
                QLabel(f"{start_str} — {end_str} UTC"))

        report_form.addRow(
            label_with_help(
                "Диапазон глубин на картах ниже:",
                "Минимальная и максимальная глубина, по которой раскрашены точки "
                "трека на картах ниже — единый диапазон для карт «до» и «после», "
                "чтобы цвета были сравнимы между собой."),
            QLabel(f"{result.get('depth_vmin', 0.0):.2f}–"
                   f"{result.get('depth_vmax', 0.0):.2f} м "
                   f"(минимум — красный, максимум — синий)"))
        report_box.setLayout(report_form)

        depth_vmin, depth_vmax = result.get("depth_vmin"), result.get("depth_vmax")
        left_view = new_map_view()
        right_view = new_map_view()

        # Мосты для синхронизации пана/зума и наведения между двумя картами —
        # каждый мост знает только про "вторую" карту (см. TrackCompareBridge).
        self.left_bridge = TrackCompareBridge(lambda: right_view)
        self.left_channel = QWebChannel()
        self.left_channel.registerObject("bridge", self.left_bridge)
        left_view.page().setWebChannel(self.left_channel)

        self.right_bridge = TrackCompareBridge(lambda: left_view)
        self.right_channel = QWebChannel()
        self.right_channel.registerObject("bridge", self.right_bridge)
        right_view.page().setWebChannel(self.right_channel)

        load_html(left_view, build_tracks_html(basemap, [
            dict(name="Эхограмма (сырой GPS)", color="red", **result["before"]["sl2"]),
            dict(name="GNSS-трек", color="blue", **result["before"]["gnss"]),
        ], depth_vmin=depth_vmin, depth_vmax=depth_vmax, sync=True), "before")
        load_html(right_view, build_tracks_html(basemap, [
            dict(name="Эхограмма (синхр.)", color="red", **result["after"]["sl2"]),
            dict(name="GNSS-трек", color="blue", **result["after"]["gnss"]),
        ], depth_vmin=depth_vmin, depth_vmax=depth_vmax, sync=True), "after")

        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("До синхронизации"))
        left_col.addWidget(left_view, 1)
        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("После синхронизации"))
        right_col.addWidget(right_view, 1)
        maps_row = QHBoxLayout()
        maps_row.addLayout(left_col, 1)
        maps_row.addLayout(right_col, 1)

        layout = QVBoxLayout()
        layout.addWidget(report_box)
        layout.addLayout(maps_row, 1)
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

        track_group = QGroupBox("Трек")
        self.track_width_spin = QSpinBox()
        self.track_width_spin.setRange(1, 20)
        self.track_width_spin.setSuffix(" px")
        self.track_width_spin.setValue(int(settings.value("track_line_width", 3)))
        track_width_row = QHBoxLayout()
        track_width_row.addWidget(QLabel("Толщина линии трека:"))
        track_width_row.addWidget(self.track_width_spin)

        self.speed_glow_spin = QSpinBox()
        self.speed_glow_spin.setRange(1, 60)
        self.speed_glow_spin.setSuffix(" px")
        self.speed_glow_spin.setValue(int(settings.value("speed_glow_width", 12)))
        speed_glow_row = QHBoxLayout()
        speed_glow_row.addWidget(QLabel("Толщина линии скорости:"))
        speed_glow_row.addWidget(self.speed_glow_spin)

        track_layout = QVBoxLayout()
        track_layout.addLayout(track_width_row)
        track_layout.addLayout(speed_glow_row)
        track_group.setLayout(track_layout)

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
        layout.addWidget(track_group)
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
        self.settings.setValue("track_line_width", self.track_width_spin.value())
        self.settings.setValue("speed_glow_width", self.speed_glow_spin.value())
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
        self.setWindowTitle(f"ОМДЖЕТ Гидро v{APP_VERSION}")
        self.setWindowIcon(QIcon(ICON_PATH))
        self.resize(1300, 1000)
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
        row1.addWidget(help_icon(SL2_FILE_HELP))
        row1.addWidget(self.sl2_edit, 1)
        row1.addWidget(sl2_btn)
        row1.addWidget(view_echogram_btn)
        row1.addWidget(sep1)
        row1.addWidget(multi_btn)
        row1.addWidget(sep1b)
        row1.addWidget(close_echograms_btn)

        self.gnss_paths = []
        self.gnss_track = None
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
        row2.addWidget(help_icon(GNSS_FILE_HELP))
        row2.addWidget(self.gnss_edit, 1)
        row2.addWidget(gnss_btn)
        row2.addWidget(sep2a)
        row2.addWidget(multi_gnss_btn)
        row2.addWidget(sep2b)
        row2.addWidget(close_gnss_btn)

        sep_hline = QFrame()
        sep_hline.setFrameShape(QFrame.Shape.HLine)
        sep_hline.setFrameShadow(QFrame.Shadow.Sunken)
        isobaths_btn = QPushButton("Построить изобаты")
        isobaths_btn.clicked.connect(self.open_isobaths)
        sep_iso = QFrame()
        sep_iso.setFrameShape(QFrame.Shape.VLine)
        sep_iso.setFrameShadow(QFrame.Shadow.Sunken)

        self.offset_btn = QPushButton("Рассчитать смещение")
        self.offset_btn.setEnabled(False)
        self.offset_btn.clicked.connect(self.calc_offset)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        settings_btn = QPushButton("Настройки")
        settings_btn.clicked.connect(self.open_settings)
        row_offset = QHBoxLayout()
        row_offset.addWidget(isobaths_btn)
        row_offset.addWidget(sep_iso)
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
        time_row = QFormLayout()
        time_row.addRow("С начала:", self.crop_start_time_spin)
        time_row.addRow("С конца:", self.crop_end_time_spin)

        self.crop_start_dist_spin = QDoubleSpinBox()
        self.crop_start_dist_spin.setRange(0.0, 1_000_000.0)
        self.crop_start_dist_spin.setSuffix(" м")
        self.crop_end_dist_spin = QDoubleSpinBox()
        self.crop_end_dist_spin.setRange(0.0, 1_000_000.0)
        self.crop_end_dist_spin.setSuffix(" м")
        dist_row = QFormLayout()
        dist_row.addRow("С начала:", self.crop_start_dist_spin)
        dist_row.addRow("С конца:", self.crop_end_dist_spin)

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

        speed_box = QGroupBox("Скорость")
        self.max_speed_label = QLabel("—")
        self.avg_speed_label = QLabel("—")
        speed_form = QFormLayout()
        speed_form.addRow("Макс.:", self.max_speed_label)
        speed_form.addRow("Средняя:", self.avg_speed_label)
        speed_box.setLayout(speed_form)

        dist_box = QGroupBox("Расстояние")
        self.track_length_label = QLabel("—")
        dist_form = QFormLayout()
        dist_form.addRow("Протяжённость:", self.track_length_label)
        dist_box.setLayout(dist_form)

        date_box = QGroupBox("Дата")
        self.start_label = QLabel("Начало: —")
        self.end_label = QLabel("Конец: —")
        self.duration_label = QLabel("Продолжительность: —")
        for lbl in (self.start_label, self.end_label, self.duration_label):
            lbl.setWordWrap(True)
        date_layout = QVBoxLayout()
        date_layout.addWidget(self.start_label)
        date_layout.addWidget(self.end_label)
        date_layout.addWidget(self.duration_label)
        date_box.setLayout(date_layout)

        summary_layout = QVBoxLayout()
        summary_layout.addWidget(date_box)
        summary_layout.addWidget(speed_box)
        summary_layout.addWidget(dist_box)
        summary_layout.addWidget(depth_box)
        summary_layout.addStretch(1)
        self.summary_panel.setLayout(summary_layout)

        self.basemap_combo = QComboBox()
        self.basemap_combo.addItems(BASEMAPS.keys())
        self.basemap_combo.currentTextChanged.connect(self.redraw_map)
        self.hide_gnss_check = QCheckBox("Скрыть GNSS трек")
        self.hide_gnss_check.toggled.connect(self.redraw_map)
        self.hide_endpoints_check = QCheckBox("Скрыть начало/конец треков")
        self.hide_endpoints_check.toggled.connect(self.redraw_map)
        self.show_speed_track_check = QCheckBox("Скорость на треке")
        self.show_speed_track_check.toggled.connect(self.redraw_map)
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(["Глубина", "Твёрдость дна", "Нет"])
        self.color_mode_combo.currentTextChanged.connect(self.redraw_map)
        sep_row3a = QFrame()
        sep_row3a.setFrameShape(QFrame.Shape.VLine)
        sep_row3a.setFrameShadow(QFrame.Shadow.Sunken)
        sep_row3a2 = QFrame()
        sep_row3a2.setFrameShape(QFrame.Shape.VLine)
        sep_row3a2.setFrameShadow(QFrame.Shadow.Sunken)
        sep_row3b = QFrame()
        sep_row3b.setFrameShape(QFrame.Shape.VLine)
        sep_row3b.setFrameShadow(QFrame.Shadow.Sunken)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Фотоподложка:"))
        row3.addWidget(self.basemap_combo, 1)
        row3.addWidget(sep_row3a)
        row3.addWidget(self.hide_gnss_check)
        row3.addWidget(sep_row3a2)
        row3.addWidget(self.hide_endpoints_check)
        row3.addWidget(self.show_speed_track_check)
        row3.addWidget(sep_row3b)
        row3.addWidget(QLabel("Раскраска:"))
        row3.addWidget(self.color_mode_combo)

        left_col = QVBoxLayout()
        left_col.addWidget(self.summary_panel)
        left_col.addWidget(crop_group)
        left_col.addWidget(offset_group)
        left_col.addStretch(1)
        left_widget = QWidget()
        left_widget.setLayout(left_col)
        left_widget.setFixedWidth(260)

        maps_row = QHBoxLayout()
        maps_row.addWidget(left_widget)
        maps_row.addWidget(self.map_view, 1)

        layout = QVBoxLayout()
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(sep_hline)
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
            self, "GNSS-трек", self.last_dir(), "GNSS-трек (*.nmea *.pos *.csv *.ubx);;Все файлы (*)")
        if path:
            self.remember_dir(path)
            self.gnss_paths = [path]
            self.gnss_edit.setText(path)
            self.update_offset_enabled()
            self.reload_gnss_track()

    def pick_multi_gnss(self):
        self.gnss_paths = []
        self._pick_next_gnss_file()

    def _pick_next_gnss_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "GNSS-трек", self.last_dir(), "GNSS-трек (*.nmea *.pos *.csv *.ubx);;Все файлы (*)")
        if not path:
            return
        self.remember_dir(path)
        self.gnss_paths.append(path)
        self.gnss_edit.setText("; ".join(self.gnss_paths))
        self.update_offset_enabled()
        self.reload_gnss_track()
        reply = QMessageBox.question(
            self, "Мультитрек", "Загрузить ещё один GNSS-трек?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._pick_next_gnss_file()

    def reload_gnss_track(self):
        """Фоновая загрузка GNSS-трека для отображения на карте (тонкой красной
        линией) — отдельно от read_gnss внутри compute_time_offset, т.к. точки
        нужны на карте сразу после загрузки файла, ещё до расчёта смещения."""
        if not self.gnss_paths:
            self.gnss_track = None
            self.redraw_map()
            return
        run_async(lambda: read_gnss_track_points(list(self.gnss_paths)),
                  on_finished=self.on_gnss_track_loaded, on_error=self.on_gnss_track_error)

    def on_gnss_track_loaded(self, track):
        self.gnss_track = track
        self.redraw_map()

    def on_gnss_track_error(self, message):
        self.gnss_track = None
        self.statusBar().showMessage(f"Не удалось прочитать GNSS-трек для показа на карте: {message}")
        self.redraw_map()

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
        self.gnss_track = None
        self.gnss_edit.setText("")
        self.update_offset_enabled()
        self.redraw_map()
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
        # Расчёт смещения работает только с одним файлом эхограммы — при
        # мультиэхограмме self.sl2_edit хранит пути через "; ", это не валидный
        # путь для read_sl2 (раньше падало необработанным OSError при клике).
        path = self.sl2_edit.text()
        self.offset_btn.setEnabled(bool(path) and ";" not in path and bool(self.gnss_paths))

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
        lat_all, lon_all, val_all, t_all, dist_all, hard_all, speed_all = [], [], [], [], [], [], []
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
            hard_all.extend(np.asarray(f["hardness"])[keep].tolist())
            speed_all.extend(np.asarray(f["speed"])[keep].tolist())
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

        self.points = dict(lat=lat_all, lon=lon_all, value=val_all, hardness=hard_all,
                            speed=speed_all, t_rel=t_all, dist=dist_all,
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

        speed = points.get("speed", [])
        if speed:
            self.max_speed_label.setText(f"{max(speed):.2f} м/с")
            self.avg_speed_label.setText(f"{sum(speed) / len(speed):.2f} м/с")
        else:
            self.max_speed_label.setText("—")
            self.avg_speed_label.setText("—")

        dist = points.get("dist", [])
        if dist:
            length_m = max(dist) - min(dist)
            self.track_length_label.setText(
                f"{length_m / 1000:.2f} км" if length_m >= 1000 else f"{length_m:.0f} м")
        else:
            self.track_length_label.setText("—")

        start, end, duration_s = points.get("start"), points.get("end"), points.get("duration_s")
        if start and end and duration_s is not None:
            start_str = datetime.fromisoformat(start).strftime("%d.%m.%Y %H:%M:%S")
            end_str = datetime.fromisoformat(end).strftime("%d.%m.%Y %H:%M:%S")
            self.start_label.setText(f"Начало: {start_str} UTC")
            self.end_label.setText(f"Конец: {end_str} UTC")
            self.duration_label.setText(f"Продолжительность: {timedelta(seconds=int(duration_s))}")
        else:
            self.start_label.setText("Начало: —")
            self.end_label.setText("Конец: —")
            self.duration_label.setText("Продолжительность: —")

    def on_echogram_error(self, message):
        self.statusBar().showMessage("Ошибка чтения файла")
        QMessageBox.critical(self, "Ошибка чтения файла", message)

    def redraw_map(self):
        points = self.points or dict(lat=[], lon=[], value=[], hardness=[], speed=[], t_rel=[], dist=[])
        mode = self.color_mode_combo.currentText()
        if mode == "Твёрдость дна":
            map_points = dict(points, value=points.get("hardness", []))
            color_by, vmin, vmax, legend_title, tooltip_suffix = True, 0.0, 1.0, "Твёрдость дна", ""
        elif mode == "Глубина":
            manual = self.settings.value("depth_mode", "auto") == "manual"
            vmin = float(self.settings.value("depth_min", 0.0)) if manual else None
            vmax = float(self.settings.value("depth_max", 10.0)) if manual else None
            map_points = points
            color_by, legend_title, tooltip_suffix = True, "Глубина, м", " м"
        else:
            map_points, color_by, vmin, vmax = points, False, None, None
            legend_title, tooltip_suffix = "", " м"
        steps = int(self.settings.value("depth_steps", 32))
        axis_mode = self.settings.value("timeline_axis", "time")
        html = build_map_html(self.basemap_combo.currentText(), map_points, color_by,
                               vmin=vmin, vmax=vmax, steps=steps, axis_mode=axis_mode,
                               legend_title=legend_title, tooltip_suffix=tooltip_suffix,
                               depth_vals=points.get("value", []),
                               speed_vals=points.get("speed", []),
                               show_endpoints=not self.hide_endpoints_check.isChecked(),
                               show_speed_track=self.show_speed_track_check.isChecked(),
                               track_width=int(self.settings.value("track_line_width", 3)),
                               speed_glow_width=int(self.settings.value("speed_glow_width", 12)),
                               gnss_track=(self.gnss_track if not self.hide_gnss_check.isChecked()
                                           else None))
        load_html(self.map_view, html, "main")

    def open_settings(self):
        dlg = SettingsDialog(self, self.settings)
        if dlg.exec():
            self.redraw_map()

    def open_isobaths(self):
        dlg = IsobathsDialog(self)
        dlg.exec()

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
        path = self.sl2_edit.text()
        if not path or ";" in path:
            QMessageBox.information(
                self, "Расчёт смещения",
                "Расчёт смещения работает только с одним файлом эхограммы (не с "
                "мультиэхограммой) — выберите один файл (кнопка «Обзор…»).")
            return
        self.offset_btn.setEnabled(False)
        self.statusBar().showMessage("Вычисление смещения…")
        run_async(lambda: compute_time_offset(path, self.gnss_paths),
                  on_finished=self.on_offset_finished, on_error=self.on_offset_error)

    def on_offset_finished(self, result):
        self.statusBar().showMessage(f"Смещение: {result['offset_s']:+.3f} с")
        self.offset_btn.setEnabled(True)
        # Дата/время из mtime файла .sl2 — лишь приближение (см. docs/CONTEXT.md);
        # синхронизация с GNSS даёт настоящее UTC-время, поэтому раз она уже
        # посчитана (только для одного загруженного файла — расчёт смещения
        # не поддерживает мультиэхограмму), обновляем сводку точным значением.
        if self.points and len(self.file_data) == 1 and result.get("start") and result.get("end"):
            self.points["start"] = result["start"]
            self.points["end"] = result["end"]
            self.points["duration_s"] = result["duration_s"]
            self.update_summary()
        dlg = OffsetDialog(self, self.basemap_combo.currentText(), result)
        dlg.exec()

    def on_offset_error(self, message):
        self.statusBar().showMessage("Ошибка вычисления смещения")
        self.offset_btn.setEnabled(True)
        QMessageBox.critical(self, "Ошибка вычисления смещения", message)


def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(ICON_PATH))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
