#!/usr/bin/env python3
"""
Главное окно sl2sync: выбор файла эхограммы, карта с точками (встроенный GPS
Lowrance) поверх выбранной онлайн-подложки, расчёт временного смещения
относительно GNSS-трека со сравнением треков до/после синхронизации.
"""
import csv
import html
import itertools
import json
import os
import re
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
from pyproj import CRS
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from PySide6.QtCore import QObject, QPoint, QRect, QSettings, QSize, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QColorDialog, QComboBox,
                                QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                                QFormLayout, QFrame, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
                                QLineEdit, QListWidget, QListWidgetItem, QMenu, QMainWindow, QMessageBox,
                                QPushButton, QRadioButton, QScrollArea, QSlider, QSpinBox,
                                QStackedWidget, QTabWidget, QTextBrowser, QTextEdit, QVBoxLayout, QWidget)

from ppk_module import PPKConfig, PPKProcessor, is_rinex, rinex_obs_info
from sl2sync import (build_parser, estimate_time_model, gnss_motion, make_proj,
                      ping_positions, read_gnss, read_sl2, read_ubx)
from sl2sync import run as sl2sync_run

APP_VERSION = "0.1.6"
APP_VERSION_DATE = "2026-09-28"          # дата этой версии — обновлять вместе с APP_VERSION
GITHUB_REPO = "andrewkena/omjet-gidro"


def _app_dir():
    """Папка с самим приложением: рядом с .exe в собранной версии (не
    временная _MEIPASS — она пересобирается при каждом запуске), рядом с
    gui.py при запуске из исходников. Здесь должны лежать файлы, которые
    пользователь правит руками (coord_systems.txt)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _bundle_dir():
    """Папка с ресурсами, вшитыми в exe (PyInstaller распаковывает их во
    временную _MEIPASS) — только для файлов на чтение (иконка)."""
    return getattr(sys, "_MEIPASS", _app_dir())


ICON_PATH = os.path.join(_bundle_dir(), "assets", "gidro.ico")

BASEMAPS = {
    "Google Спутник": dict(
        url="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        max_zoom=20),
    "Google Карта": dict(
        url="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        max_zoom=20),
}

# Системы координат для колонок E/N и грида на вкладке «Экспорт данных».
# None — автоподбор зоны UTM по WGS84 (текущее поведение sl2sync по умолчанию).
# Местные СК (МСК) — параметры получены от пользователя из mapinfow.prj
# (формат: "имя", 8, 1001, 7, центр.меридиан, 0, 1, ложное_смещение_E, ложное_смещение_N;
# 8 = Gauss-Kruger/tmerc, датум 1001 = Пулково-1942/эллипсоид Красовского).
# towgs84 — общероссийские параметры перехода Пулково-1942 → WGS84 (точность
# порядка метра, региональные уточнения по областям не заданы).
_RU_TOWGS84 = "23.92,-141.27,-80.9,0,0.35,0.82,-0.12"


def _gk_proj4(lon0, x0, y0):
    """proj4 для местной СК типа МСК (Гаусс-Крюгер/tmerc, Красовский) —
    см. _RU_TOWGS84 выше."""
    return (f"+proj=tmerc +lat_0=0 +lon_0={lon0} +k=1 +x_0={x0} +y_0={y0} "
            f"+ellps=krass +towgs84={_RU_TOWGS84} +units=m +no_defs")


COORD_SYSTEMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coord_systems.txt")
_COORD_SYSTEMS_ROW_RE = re.compile(
    r'^"([^"]+)"\s*,\s*([\-\d.]+)\s*,\s*([\-\d.]+)\s*,\s*([\-\d.]+)\s*,\s*'
    r'([\-\d.]+)\s*,\s*([\-\d.]+)\s*,\s*([\-\d.]+)\s*,\s*([\-\d.]+)\s*,\s*([\-\d.]+)\s*$')


def load_coord_systems_file(path=COORD_SYSTEMS_FILE):
    """Читает местные СК из текстового файла в формате mapinfow.prj (см.
    коммент в самом файле) — пользователь может дописывать туда новые системы
    руками, без правки кода. Строка, не подходящая под формат записи СК,
    считается заголовком раздела и добавляется в название следующих за ней
    систем (до следующего заголовка) — так исходные блоки текста можно
    вставлять в файл как есть, без переформатирования."""
    systems = {}
    if not os.path.exists(path):
        return systems
    section = ""
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = _COORD_SYSTEMS_ROW_RE.match(line)
            if not m:
                section = line
                continue
            name, proj_id, _datum, _unit, lon0, _lat0, _k, x0, y0 = m.groups()
            if int(float(proj_id)) != 8:
                continue  # поддерживается только Gauss-Kruger/tmerc (id=8)
            label = f"{section} — {name}" if section else name
            systems[label] = _gk_proj4(float(lon0), float(x0), float(y0))
    return systems


COORD_SYSTEMS = {"WGS84 / UTM (авто, по данным)": None, **load_coord_systems_file()}


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


def warning_icon(size=16):
    """Красный кружок с «!» — значок для элементов списка (QListWidgetItem
    поддерживает только QIcon, не произвольный виджет/стиль, как help_icon)."""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#cc3333"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(0, 0, size, size)
    painter.setPen(QColor("white"))
    font = painter.font()
    font.setBold(True)
    font.setPixelSize(int(size * 0.7))
    painter.setFont(font)
    painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "!")
    painter.end()
    return QIcon(pix)


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
    "приёмника (обычно одиночный, без поправок в реальном времени). Постобработку\n"
    "удобнее делать на вкладке «PPK» — там свои поля для ровера и базы\n"
    "(можно сразу несколько файлов), результат можно подставить сюда одной кнопкой."
)
GNSS_BASE_FILE_HELP = (
    "Запись базовой (стационарной) GNSS-станции — файл .ubx (сырой поток) либо\n"
    "уже готовый RINEX OBS (.obs/.rnx или годовой .NNo) — например, после конвертации\n"
    "фирменным ПО приёмника: EFT M4 конвертируют программой EFT PP, ComNav — Compass.\n"
    "Нужна только для постобработки PPK. Для обычного расчёта смещения/карты/изобат\n"
    "не используется. Для загрузки нескольких роверов/баз сразу удобнее вкладка\n"
    "«PPK» — там свои, независимые от этого поля, списки файлов."
)
WATERLINE_FILE_HELP = (
    "Точки уреза/русла — CSV с колонками lat/lon (разделитель определяется\n"
    "автоматически), обычный текст (на каждой строке два числа lat lon — все\n"
    "такие точки считаются урезом) либо KML: цвет линии определяет тип —\n"
    "оранжевая/красная — урез, голубая/синяя — русло; линии без стиля тоже\n"
    "считаются урезом."
)

PPK_FILES_HELP = (
    "Принимаются:\n"
    "• .ubx — сырой поток u-blox, конвертируется в RINEX автоматически (convbin).\n"
    "• Уже готовый RINEX OBS — .obs/.rnx или годовой .NNo (например .24o) с парным\n"
    "  NAV-файлом (.NNn/.NNp) рядом — используется как есть, без конвертации. Так\n"
    "  выглядит результат конвертации фирменным ПО приёмника — свой формат сырых\n"
    "  данных и своя программа конвертации в RINEX почти у каждого производителя\n"
    "  (например, EFT M4 → EFT PP, ComNav → Compass); RTKLIB/convbin эти сырые\n"
    "  форматы не читает, только уже готовый RINEX от них.\n"
    "• RINEX (и его NAV-файл рядом) можно сжатым в .gz — например, скачанный с\n"
    "  постоянной станции (CORS/IGS) — распаковывается автоматически."
)
PPK_SYSTEMS_HELP = (
    "Спутниковые системы, участвующие в расчёте (буквы): G — GPS, R — ГЛОНАСС,\n"
    "E — Galileo, C — BeiDou. По умолчанию GREC — все четыре. Меньше систем —\n"
    "меньше спутников в решении (хуже геометрия), но иногда полезно исключить\n"
    "систему с плохим качеством данных у конкретного приёмника."
)
PPK_ELMASK_HELP = (
    "Маска возвышения — минимальный угол спутника над горизонтом (°), ниже\n"
    "которого измерения отбрасываются. Сигналы от низких спутников идут через\n"
    "более длинный слой атмосферы и чаще содержат многолучевость (отражения от\n"
    "построек, воды, склонов). Типично 10–15°; в застройке/лесу — 20–25°."
)
PPK_AR_RATIO_HELP = (
    "Порог теста отношения (ratio test) для фиксации целочисленных\n"
    "неоднозначностей фазы (Ambiguity Resolution) — чем выше, тем строже\n"
    "критерий: реже ложный фикс, но и сам фикс происходит реже/дольше.\n"
    "Типичные значения: 3 (мягкий), 5 (стандартный), 10 (строгий)."
)
PPK_AR_MODE_HELP = (
    "Режим разрешения неоднозначностей фазы:\n"
    "• Continuous — фикс пересчитывается заново на каждой эпохе (обычный режим\n"
    "  для кинематики, подходит по умолчанию).\n"
    "• Fix and hold — однажды зафиксированное значение «удерживается» как\n"
    "  жёсткое ограничение для последующих эпох: может ускорить возврат к\n"
    "  фиксу после срыва, но ошибочная фиксация тоже «держится» дольше."
)
PPK_GLO_AR_HELP = (
    "Учёт межканальных смещений ГЛОНАСС (FDMA — разные спутники на разных\n"
    "частотах) при фиксации неоднозначностей:\n"
    "• Вкл. — ровер и база считаются одинаковыми/совместимыми приёмниками\n"
    "  (смещения взаимно уничтожаются) — обычный выбор, если оба приёмника\n"
    "  одной модели.\n"
    "• Autocal — смещения оцениваются автоматически по самим данным.\n"
    "• Выкл. — неоднозначности ГЛОНАСС не фиксируются (сигнал используется\n"
    "  только как дополнительное псевдодальномерное измерение)."
)
PPK_FREQUENCY_HELP = (
    "Несущие частоты в расчёте:\n"
    "• L1 — одночастотное решение, проще, но менее точно и устойчиво,\n"
    "  особенно на длинных базовых линиях (ровер далеко от базы).\n"
    "• L1+L2 — двухчастотное решение, заметно точнее и надёжнее (позволяет\n"
    "  скорректировать ионосферную задержку) — выбирайте, если оба приёмника\n"
    "  и сырые данные это поддерживают (см. «Подробности» — GPS[..]: L1 L2)."
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


def build_ppk_map_html(basemap, rover_name, rover_lat, rover_lon, base_name, base_lat, base_lon):
    """Мини-карта результата PPK: трек ровера (зелёная линия, по решению RTKLIB)
    и флажок на месте базы — обе подписаны именами файлов. Тот же цветовой код
    (зелёный/красный), что и на FileTimelineWidget. Если подписи начала трека
    ровера и флажка базы попадают на одно и то же место экрана (типично —
    ровер стартует рядом с базой), обе подписи разводятся в стороны и
    соединяются с истинной точкой тонкой пунктирной «полкой»-выноской."""
    tile = BASEMAPS[basemap]
    all_lat, all_lon = list(rover_lat), list(rover_lon)
    if base_lat is not None and base_lon is not None:
        all_lat.append(base_lat)
        all_lon.append(base_lon)
    center = [sum(all_lat) / len(all_lat), sum(all_lon) / len(all_lon)] if all_lat else [0, 0]

    lines = []
    coords = list(zip(rover_lat, rover_lon))
    rover_start_js = json.dumps(coords[0]) if coords else "null"
    if coords:
        lines.append(f"L.polyline({json.dumps(coords)}, {{color: '#2ecc71', weight: 3}})"
                     f".bindTooltip({json.dumps(rover_name)}).addTo(map);")
    base_js = json.dumps([base_lat, base_lon]) if base_lat is not None and base_lon is not None else "null"

    bounds = list(zip(all_lat, all_lon))
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}}</style>
</head><body>
<div id="map"></div>
<script>
var map = L.map('map', {{preferCanvas: true, attributionControl: false}})
  .setView([{center[0]}, {center[1]}], 15);
L.tileLayer('{tile["url"]}', {{maxZoom: {tile["max_zoom"]}}}).addTo(map);
{chr(10).join(lines)}
var bounds = {json.dumps(bounds)};
if (bounds.length > 1) {{ map.fitBounds(bounds, {{padding: [20, 20]}}); }}

var roverStart = {rover_start_js};
var basePt = {base_js};
var roverName = {json.dumps(rover_name)};
var baseName = {json.dumps(base_name)};

function addLabeledPoint(latlng, color, name, isFlag, otherLatLng) {{
  var shift = null;
  if (otherLatLng) {{
    var p1 = map.latLngToContainerPoint(latlng);
    var p2 = map.latLngToContainerPoint(otherLatLng);
    var dx = p1.x - p2.x, dy = p1.y - p2.y;
    if (Math.sqrt(dx * dx + dy * dy) < 50) {{
      shift = isFlag ? [36, -36] : [-36, 36];
    }}
  }}
  var labelLatLng = latlng;
  if (shift) {{
    var p = map.latLngToContainerPoint(latlng);
    labelLatLng = map.containerPointToLatLng([p.x + shift[0], p.y + shift[1]]);
    L.polyline([latlng, labelLatLng], {{color: color, weight: 1, dashArray: '3,3'}}).addTo(map);
    L.circleMarker(latlng, {{radius: 3, color: color, fillColor: color, fillOpacity: 1}}).addTo(map);
  }}
  if (isFlag) {{
    var flagIcon = L.divIcon({{html: '<div style="font-size:22px;line-height:22px">\U0001F6A9</div>',
                              className: '', iconSize: [22, 22], iconAnchor: [2, 20]}});
    L.marker(labelLatLng, {{icon: flagIcon}})
      .bindTooltip(name, {{permanent: true, direction: 'right', offset: [4, -10]}}).addTo(map);
  }} else {{
    L.circleMarker(labelLatLng, {{radius: 5, weight: 2, color: '#fff', fillColor: color, fillOpacity: 1}})
      .bindTooltip(name, {{permanent: true, direction: 'left', offset: [-4, -6]}}).addTo(map);
  }}
}}

if (roverStart) {{ addLabeledPoint(roverStart, '#2ecc71', roverName, false, basePt); }}
if (basePt) {{ addLabeledPoint(basePt, '#e74c3c', baseName, true, roverStart); }}
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
                lat=s["lat"][m].tolist(), lon=s["lon"][m].tolist(),
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


def smooth_polyline_corners(points, iterations=2):
    """Сглаживает острые углы нарисованной вручную линии (Chaikin corner-cutting):
    начальная и конечная точки линии остаются на месте, внутренние углы —
    скругляются. Меньше 3 точек — сглаживать нечего, возвращает как есть."""
    pts = list(points)
    if len(pts) < 3:
        return pts
    for _ in range(iterations):
        smoothed = [pts[0]]
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            smoothed.extend([q, r])
        smoothed.append(pts[-1])
        pts = smoothed
    return pts


def _densify_polyline(xs, ys, step):
    """Добавляет промежуточные точки вдоль ломаной (xs, ys — координаты в
    проекции, метры) через каждые `step` — редкие клики мышью иначе оставляют
    разрывы там, где нужна плотная опора (глубина=0 по всему урезу, а не
    только в точках клика; «принудительная вода» вдоль русла)."""
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if len(xs) < 2:
        return xs, ys
    out_x, out_y = [xs[0]], [ys[0]]
    for i in range(len(xs) - 1):
        seg_len = float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]))
        steps = max(1, int(seg_len / step))
        t = np.linspace(0.0, 1.0, steps + 1)[1:]
        out_x.extend((xs[i] + t * (xs[i + 1] - xs[i])).tolist())
        out_y.extend((ys[i] + t * (ys[i + 1] - ys[i])).tolist())
    return np.array(out_x), np.array(out_y)


def _build_terrain_mesh(GX, GY, Z, max_dim=120):
    """Уменьшает грид глубин до разумного размера (для передачи в браузер и
    отрисовки в Three.js — вкладка «3D дно») и переводит в локальные метры
    относительно угла грида. Маскированные (NaN) ячейки остаются null — в
    3D-сетке на их месте будет дырка, как и пустая область на 2D карте."""
    ny, nx = Z.shape
    stride = max(1, int(np.ceil(max(nx, ny) / max_dim)))
    Zs = Z[::stride, ::stride]
    Xs = GX[::stride, ::stride]
    Ys = GY[::stride, ::stride]
    x0, y0 = float(Xs[0, 0]), float(Ys[0, 0])
    cell_x = float(Xs[0, 1] - Xs[0, 0])
    cell_y = float(Ys[1, 0] - Ys[0, 0])
    z_rows = [[(None if not np.isfinite(v) else round(float(v), 3)) for v in row] for row in Zs]
    finite = Zs[np.isfinite(Zs)]
    zmin = float(finite.min()) if finite.size else 0.0
    zmax = float(finite.max()) if finite.size else 1.0
    return dict(nx=int(Zs.shape[1]), ny=int(Zs.shape[0]), x0=x0, y0=y0,
                cellX=cell_x, cellY=cell_y, z=z_rows, zmin=zmin, zmax=zmax)


def compute_isobaths(lat, lon, depth, waterline_lines, cell, interval, fill_steps, smooth=0.0,
                      channel_lines=None):
    """Грид глубин (линейная интерполяция) + линии постоянной глубины и залитые
    диапазоны глубин через matplotlib.contour/contourf. Точки линий уреза воды
    (нарисованных вручную, глубина 0) подмешиваются к данным эхолота — это
    стандартный приём в батиметрии: сонар не измеряет вплотную к берегу,
    а урез задаёт границу 0 м. Линий уреза (`waterline_lines`) и линий русла
    (`channel_lines`) может быть несколько — каждый параметр это список линий,
    линия это список точек (lat, lon). Точки русла — тоже вспомогательные, но
    глубина им не назначается фиксированной: берётся у ближайшего реального
    промера, чтобы просто направить интерполяцию вдоль русла там, где
    промеров мало, без выдумывания глубины.

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
    from scipy.spatial import cKDTree

    waterline_lines = waterline_lines or []
    channel_lines = channel_lines or []
    channel_points = [p for line in channel_lines for p in line]
    lat_list, lon_list, depth_list = list(lat), list(lon), list(depth)

    crs, fwd, inv = make_proj(float(np.median(lon_list)), float(np.median(lat_list)))
    track_E, track_N = fwd.transform(lon_list, lat_list)

    channel_depth = []
    ch_E = ch_N = np.array([])
    if channel_points:
        tree = cKDTree(np.column_stack([track_E, track_N]))
        ch_E, ch_N = fwd.transform([p[1] for p in channel_points], [p[0] for p in channel_points])
        _, idx = tree.query(np.column_stack([ch_E, ch_N]))
        channel_depth = [float(depth_list[i]) for i in idx]

    # Глубина 0 должна держаться по всей линии уреза, а не только в точках
    # клика мышью — иначе между редкими кликами линейная интерполяция тянет
    # значение от ближайшего реального промера, и глубина у самой кромки воды
    # не подходит к нулю. Уплотняем каждую линию уреза с шагом в половину
    # ячейки сетки перед тем, как отдать её в griddata как точки глубины 0.
    wl_dense_E, wl_dense_N = [], []
    for line in waterline_lines:
        if len(line) < 2:
            if line:
                e0, n0 = fwd.transform([line[0][1]], [line[0][0]])
                wl_dense_E.append(float(e0[0]))
                wl_dense_N.append(float(n0[0]))
            continue
        le, ln = fwd.transform([p[1] for p in line], [p[0] for p in line])
        de, dn = _densify_polyline(le, ln, cell * 0.5)
        wl_dense_E.extend(de.tolist())
        wl_dense_N.extend(dn.tolist())

    E = np.concatenate([track_E, np.array(wl_dense_E), ch_E])
    N = np.concatenate([track_N, np.array(wl_dense_N), ch_N])
    depth_arr = np.array(depth_list + [0.0] * len(wl_dense_E) + channel_depth)

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
    # method="linear" не считает ничего за выпуклой оболочкой точек — без
    # уреза эти ячейки честно остаются пустыми (nan_mask ниже), но если урез
    # нарисован, он сам определяет, где должна быть вода (см. water_mask
    # ниже), и тогда такие ячейки внутри уреза дозаполняются ближайшим
    # соседом — иначе изобаты не доходят до линии уреза в заливах/бухтах,
    # куда лодка не подходила вплотную к берегу. Дозаполняем всегда (не
    # только внутри будущей маски), чтобы gaussian_filter ниже не размывал
    # NaN на соседние валидные ячейки.
    nan_mask = np.isnan(Z)
    if nan_mask.any():
        Zfill = griddata((E, N), depth_arr, (GX, GY), method="nearest")
        Z = np.where(nan_mask, Zfill, Z)
    if smooth and smooth > 0:
        Z = gaussian_filter(Z, sigma=float(smooth))

    # Изобаты не должны заходить на сушу за нарисованный урез воды: каждую
    # линию уреза с >=3 точками замыкаем в многоугольник и определяем, какая
    # сторона — вода, по тому, где лежит сам трек (лодка не плавает по суше);
    # ячейки сетки на другой стороне маскируем — тогда contour/contourf там
    # ничего не рисуют, без ручного отреза геометрии. Несколько линий уреза
    # (например, оба берега узкого залива) сужают область воды последовательно
    # (пересечение масок).
    #
    # Урез почти всегда рисуют как ОТКРЫТУЮ линию вдоль берега (не замкнутый
    # контур острова/озера целиком) — a Path.contains_points всё равно неявно
    # замыкает контур прямым отрезком от последней точки к первой. Если этот
    # отрезок просто прямая через всю акваторию, он отрезает область воды по
    # диагонали безо всякой связи с нарисованной линией (видно на карте как
    # залив, залитый лишь до случайной прямой). Поэтому открытую линию перед
    # замыканием продлеваем по касательной на обоих концах далеко за пределы
    # сетки — тогда неявное замыкающее ребро проходит вне видимой области, а
    # маску формирует сама нарисованная линия (продолженная в сторону, куда
    # она и так шла), а не срез напрямик. Уже замкнутый контур (последняя
    # точка рядом с первой — остров, изолированная акватория) продлевать не
    # нужно, используем как есть.
    span = float(np.hypot(E.max() - E.min(), N.max() - N.min())) * 2.0
    water_mask = None
    for line in waterline_lines:
        if len(line) < 3:
            continue
        wl_E, wl_N = fwd.transform([p[1] for p in line], [p[0] for p in line])
        closed = np.hypot(wl_E[0] - wl_E[-1], wl_N[0] - wl_N[-1]) < max(cell * 3, span * 0.01)
        if not closed and span > 0:
            d0 = np.hypot(wl_E[0] - wl_E[1], wl_N[0] - wl_N[1]) or 1.0
            p_start = (wl_E[0] + (wl_E[0] - wl_E[1]) / d0 * span,
                       wl_N[0] + (wl_N[0] - wl_N[1]) / d0 * span)
            d1 = np.hypot(wl_E[-1] - wl_E[-2], wl_N[-1] - wl_N[-2]) or 1.0
            p_end = (wl_E[-1] + (wl_E[-1] - wl_E[-2]) / d1 * span,
                     wl_N[-1] + (wl_N[-1] - wl_N[-2]) / d1 * span)
            wl_E = np.concatenate([[p_start[0]], wl_E, [p_end[0]]])
            wl_N = np.concatenate([[p_start[1]], wl_N, [p_end[1]]])
        wl_path = MplPath(np.column_stack([wl_E, wl_N]))
        track_pts = np.column_stack([E[:len(lat)], N[:len(lat)]])
        track_inside = wl_path.contains_points(track_pts).mean() >= 0.5
        grid_inside = wl_path.contains_points(
            np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
        line_mask = grid_inside if track_inside else ~grid_inside
        water_mask = line_mask if water_mask is None else (water_mask & line_mask)

    if water_mask is not None and channel_lines:
        # Русло — тоже точно вода, даже если нарисованный урез (например,
        # только у одного берега) формально не накрывает эту область: полоса
        # в пару ячеек сетки вокруг каждой линии русла принудительно
        # включается в воду, иначе изобаты обрываются, не доходя до русла.
        dense_E, dense_N = [], []
        for line in channel_lines:
            if len(line) < 2:
                continue
            le, ln = fwd.transform([p[1] for p in line], [p[0] for p in line])
            de, dn = _densify_polyline(le, ln, cell * 0.5)
            dense_E.extend(de.tolist())
            dense_N.extend(dn.tolist())
        if dense_E:
            tree = cKDTree(np.column_stack([dense_E, dense_N]))
            dist, _ = tree.query(np.column_stack([GX.ravel(), GY.ravel()]))
            channel_forced = dist.reshape(GX.shape) < cell * 2.0
            water_mask = water_mask | channel_forced

    if water_mask is not None:
        # Урез нарисован — он единственная граница, есть данные или нет.
        Z = np.where(water_mask, Z, np.nan)
    else:
        # Уреза нет — не выдумываем данные за пределами того, что реально
        # накрывает линейная интерполяция (прежнее поведение).
        Z = np.where(nan_mask, np.nan, Z)

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
    terrain = _build_terrain_mesh(GX, GY, Z)
    return dict(lines=lines, bands=bands, zmin=zmin, zmax=zmax, terrain=terrain)


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


def read_waterline_kml(path):
    """Читает линии уреза/русла из KML (LineString/MultiGeometry). Тип линии
    определяет цвет стиля (свой у Placemark либо через styleUrl на <Style
    id=...> уровня документа) — KML хранит цвет как aabbggrr (не rgb):
    красный/оранжевый канал (r>b) — урез, синий/голубой (b>r) — русло. Линии
    без определённого цвета (нет стиля вовсе) считаются урезом — для
    совместимости с простыми KML без оформления. Каждый блок <coordinates>
    (Placemark может содержать несколько — например, в MultiGeometry) — своя
    отдельная линия. Возвращает (urez_lines, channel_lines) — списки линий,
    линия — список точек (lat, lon)."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ValueError(f"Некорректный KML: {e}")

    m = re.match(r"\{(.+)\}", root.tag)
    uri = m.group(1) if m else None

    def tag(name):
        return f"{{{uri}}}{name}" if uri else name

    style_colors = {}
    for style in root.iter(tag("Style")):
        sid = style.get("id")
        color_el = style.find(f"{tag('LineStyle')}/{tag('color')}")
        if sid and color_el is not None and color_el.text:
            style_colors[sid] = color_el.text.strip()

    urez_lines, channel_lines = [], []
    for placemark in root.iter(tag("Placemark")):
        color_hex = None
        inline_color = placemark.find(f"{tag('Style')}/{tag('LineStyle')}/{tag('color')}")
        if inline_color is not None and inline_color.text:
            color_hex = inline_color.text.strip()
        else:
            style_url = placemark.find(tag("styleUrl"))
            if style_url is not None and style_url.text:
                color_hex = style_colors.get(style_url.text.strip().lstrip("#"))

        kind = "urez"
        if color_hex and len(color_hex) == 8:
            try:
                r, b = int(color_hex[6:8], 16), int(color_hex[2:4], 16)
                if b > r:
                    kind = "channel"
            except ValueError:
                pass

        for coord_el in placemark.iter(tag("coordinates")):
            if not coord_el.text:
                continue
            line = []
            for tuple_str in coord_el.text.split():
                parts = tuple_str.split(",")
                if len(parts) < 2:
                    continue
                try:
                    lon, lat = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                line.append((lat, lon))
            if line:
                (urez_lines if kind == "urez" else channel_lines).append(line)

    if not urez_lines and not channel_lines:
        raise ValueError("Не удалось прочитать линии из KML-файла")
    return urez_lines, channel_lines


def build_isobaths_map_html(basemap, lat, lon, excluded_indices=None):
    tile = BASEMAPS[basemap]
    center = [sum(lat) / len(lat), sum(lon) / len(lon)] if lat else [0, 0]
    pts = list(zip(lat, lon))
    excluded_json = json.dumps(sorted(excluded_indices) if excluded_indices else [])
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
var excludedIdx = new Set({excluded_json});
var ptsLayer = L.featureGroup();
var ptsMarkers = pts.map(function (p, i) {{
  var m = L.circleMarker([p[0], p[1]], {{radius: 2, weight: 0, fillColor: '#ff0000',
                                         fillOpacity: 0.7, interactive: false}});
  if (!excludedIdx.has(i)) {{ m.addTo(ptsLayer); }}
  return m;
}});
ptsLayer.addTo(map);
if (pts.length) {{ map.fitBounds(ptsLayer.getBounds()); }}

function setTrackVisible(v) {{
  if (v) {{ if (!map.hasLayer(ptsLayer)) {{ map.addLayer(ptsLayer); }} }}
  else {{ if (map.hasLayer(ptsLayer)) {{ map.removeLayer(ptsLayer); }} }}
}}

// Очистка точек: режим «Удалить точки» — клик по ближайшей точке трека
// переключает её исключение из расчёта изобат (клик ещё раз — вернуть).
var cleanMode = false;
function setCleanMode(v) {{ cleanMode = v; updateCursor(); }}
function nearestPointIdx(latlng) {{
  var bestI = -1, bestD = 12;
  for (var i = 0; i < pts.length; i++) {{
    var d = distPx(latlng, L.latLng(pts[i][0], pts[i][1]));
    if (d < bestD) {{ bestD = d; bestI = i; }}
  }}
  return bestI;
}}
function restoreAllPoints() {{
  excludedIdx.forEach(function (i) {{ ptsMarkers[i].addTo(ptsLayer); }});
  excludedIdx.clear();
}}
map.on('click', function (e) {{
  if (!cleanMode) {{ return; }}
  var i = nearestPointIdx(e.latlng);
  if (i < 0) {{ return; }}
  if (excludedIdx.has(i)) {{ excludedIdx.delete(i); ptsMarkers[i].addTo(ptsLayer); }}
  else {{ excludedIdx.add(i); ptsLayer.removeLayer(ptsMarkers[i]); }}
  if (bridge) {{ bridge.pointToggled(i); }}
}});

// Редактор линий уреза/русла: несколько линий каждого типа, правая кнопка
// мыши завершает текущую линию (или удаляет линию под курсором, если сейчас
// ничего не рисуется), Ctrl+клик вставляет точку в ближайшую линию, Shift+
// клик удаляет ближайшую точку, концы линий (своей и чужого типа) притягивают
// друг друга при постановке новой точки.
var SNAP_PX = 15, HIT_PX = 10;
var drawMode = false, drawChannelMode = false;
var waterlineLines = [], channelLines = [];
var currentLine = [], currentLineType = null;

var waterlineLinesLayer = L.layerGroup();
var waterlineMarkersLayer = L.layerGroup();
var channelLinesLayer = L.layerGroup();
var channelMarkersLayer = L.layerGroup();
var waterlineGroup = L.layerGroup(
  [waterlineLinesLayer, waterlineMarkersLayer, channelLinesLayer, channelMarkersLayer]).addTo(map);

var smoothPreview = false;
function chaikinSmooth(points, iterations) {{
  var pts = points.slice();
  if (pts.length < 3) {{ return pts; }}
  for (var it = 0; it < iterations; it++) {{
    var smoothed = [pts[0]];
    for (var i = 0; i < pts.length - 1; i++) {{
      var p0 = pts[i], p1 = pts[i + 1];
      smoothed.push([0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]],
                     [0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]]);
    }}
    smoothed.push(pts[pts.length - 1]);
    pts = smoothed;
  }}
  return pts;
}}
function setSmoothPreview(v) {{ smoothPreview = v; redrawAll(); }}

function drawLineSet(lines, current, color, linesLayer, markersLayer) {{
  linesLayer.clearLayers();
  markersLayer.clearLayers();
  lines.forEach(function (line) {{
    // Линия на экране — сглаженная (если включено), но вершины-маркеры всегда
    // по исходным точкам: так видно и куда реально кликал пользователь (для
    // Ctrl/Shift-редактирования), и как линия ляжет в расчёте.
    var renderLine = smoothPreview ? chaikinSmooth(line, 2) : line;
    if (renderLine.length > 1) {{
      L.polyline(renderLine, {{color: color, weight: 3, interactive: false}}).addTo(linesLayer);
    }}
    line.forEach(function (p) {{
      L.circleMarker(p, {{radius: 3, color: color, fillColor: color, fillOpacity: 1,
                          interactive: false}}).addTo(markersLayer);
    }});
  }});
  if (current && current.length) {{
    var renderCurrent = smoothPreview ? chaikinSmooth(current, 2) : current;
    if (renderCurrent.length > 1) {{
      L.polyline(renderCurrent, {{color: color, weight: 3, interactive: false}}).addTo(linesLayer);
    }}
    current.forEach(function (p) {{
      L.circleMarker(p, {{radius: 3, color: color, fillColor: '#ffffff', fillOpacity: 1,
                          interactive: false}}).addTo(markersLayer);
    }});
  }}
}}
function redrawAll() {{
  drawLineSet(waterlineLines, currentLineType === 'waterline' ? currentLine : null,
              '#ffa500', waterlineLinesLayer, waterlineMarkersLayer);
  drawLineSet(channelLines, currentLineType === 'channel' ? currentLine : null,
              '#00bfff', channelLinesLayer, channelMarkersLayer);
}}

var bridge = null;
new QWebChannel(qt.webChannelTransport, function (channel) {{ bridge = channel.objects.bridge; }});

function sendState() {{
  if (!bridge) {{ return; }}
  var wl = waterlineLines.slice(), ch = channelLines.slice();
  if (currentLine.length) {{
    if (currentLineType === 'waterline') {{ wl = wl.concat([currentLine]); }}
    else if (currentLineType === 'channel') {{ ch = ch.concat([currentLine]); }}
  }}
  bridge.linesChanged(JSON.stringify({{waterline: wl, channel: ch}}));
}}

function updateCursor() {{
  map.getContainer().style.cursor = (drawMode || drawChannelMode || cleanMode) ? 'crosshair' : '';
}}

function finishCurrentLine() {{
  if (currentLine.length > 0) {{
    if (currentLineType === 'waterline') {{ waterlineLines.push(currentLine); }}
    else if (currentLineType === 'channel') {{ channelLines.push(currentLine); }}
  }}
  currentLine = [];
  redrawAll();
  sendState();
}}

function setDrawMode(v) {{
  if (v) {{
    if (currentLine.length && currentLineType !== 'waterline') {{ finishCurrentLine(); }}
    drawMode = true; currentLineType = 'waterline';
  }} else {{
    drawMode = false;
    if (currentLineType === 'waterline') {{ finishCurrentLine(); }}
  }}
  updateCursor();
}}
function setDrawChannelMode(v) {{
  if (v) {{
    if (currentLine.length && currentLineType !== 'channel') {{ finishCurrentLine(); }}
    drawChannelMode = true; currentLineType = 'channel';
  }} else {{
    drawChannelMode = false;
    if (currentLineType === 'channel') {{ finishCurrentLine(); }}
  }}
  updateCursor();
}}
function addLines(payload) {{
  (payload.waterline || []).forEach(function (line) {{ if (line.length) {{ waterlineLines.push(line); }} }});
  (payload.channel || []).forEach(function (line) {{ if (line.length) {{ channelLines.push(line); }} }});
  redrawAll();
}}
function setWaterlineVisible(v) {{ setLayerVisible(waterlineGroup, v); }}

function distPx(a, b) {{ return map.latLngToContainerPoint(a).distanceTo(map.latLngToContainerPoint(b)); }}
function segDistPx(p, a, b) {{
  var pp = map.latLngToContainerPoint(p), pa = map.latLngToContainerPoint(a), pb = map.latLngToContainerPoint(b);
  var dx = pb.x - pa.x, dy = pb.y - pa.y, len2 = dx * dx + dy * dy;
  var t = len2 ? Math.max(0, Math.min(1, ((pp.x - pa.x) * dx + (pp.y - pa.y) * dy) / len2)) : 0;
  var ddx = pp.x - (pa.x + t * dx), ddy = pp.y - (pa.y + t * dy);
  return Math.sqrt(ddx * ddx + ddy * ddy);
}}
function snapToEndpoint(latlng) {{
  var best = null, bestD = SNAP_PX;
  function check(p) {{
    var d = distPx(latlng, L.latLng(p[0], p[1]));
    if (d < bestD) {{ bestD = d; best = p; }}
  }}
  waterlineLines.forEach(function (l) {{ if (l.length) {{ check(l[0]); if (l.length > 1) {{ check(l[l.length - 1]); }} }} }});
  channelLines.forEach(function (l) {{ if (l.length) {{ check(l[0]); if (l.length > 1) {{ check(l[l.length - 1]); }} }} }});
  return best ? L.latLng(best[0], best[1]) : latlng;
}}
function findLineNear(latlng) {{
  var best = null, bestD = HIT_PX;
  function scan(lines, type) {{
    lines.forEach(function (line, idx) {{
      for (var i = 0; i < line.length - 1; i++) {{
        var d = segDistPx(latlng, L.latLng(line[i][0], line[i][1]), L.latLng(line[i + 1][0], line[i + 1][1]));
        if (d < bestD) {{ bestD = d; best = {{type: type, index: idx}}; }}
      }}
      if (line.length === 1 && distPx(latlng, L.latLng(line[0][0], line[0][1])) < bestD) {{
        bestD = distPx(latlng, L.latLng(line[0][0], line[0][1])); best = {{type: type, index: idx}};
      }}
    }});
  }}
  scan(waterlineLines, 'waterline');
  scan(channelLines, 'channel');
  return best;
}}
function deleteLine(hit) {{
  var arr = hit.type === 'waterline' ? waterlineLines : channelLines;
  arr.splice(hit.index, 1);
  redrawAll();
  sendState();
}}
function insertPointIntoNearestLine(latlng) {{
  var best = null, bestD = HIT_PX;
  function scan(lines, type) {{
    lines.forEach(function (line, li) {{
      for (var i = 0; i < line.length - 1; i++) {{
        var d = segDistPx(latlng, L.latLng(line[i][0], line[i][1]), L.latLng(line[i + 1][0], line[i + 1][1]));
        if (d < bestD) {{ bestD = d; best = {{type: type, li: li, seg: i, current: false}}; }}
      }}
    }});
  }}
  scan(waterlineLines, 'waterline');
  scan(channelLines, 'channel');
  for (var i = 0; i < currentLine.length - 1; i++) {{
    var d = segDistPx(latlng, L.latLng(currentLine[i][0], currentLine[i][1]), L.latLng(currentLine[i + 1][0], currentLine[i + 1][1]));
    if (d < bestD) {{ bestD = d; best = {{current: true, seg: i}}; }}
  }}
  if (!best) {{ return; }}
  var newPt = [latlng.lat, latlng.lng];
  if (best.current) {{ currentLine.splice(best.seg + 1, 0, newPt); }}
  else {{ (best.type === 'waterline' ? waterlineLines : channelLines)[best.li].splice(best.seg + 1, 0, newPt); }}
  redrawAll();
  sendState();
}}
function deleteNearestPoint(latlng) {{
  var best = null, bestD = HIT_PX;
  function scan(lines, type) {{
    lines.forEach(function (line, li) {{
      line.forEach(function (p, pi) {{
        var d = distPx(latlng, L.latLng(p[0], p[1]));
        if (d < bestD) {{ bestD = d; best = {{type: type, li: li, pi: pi, current: false}}; }}
      }});
    }});
  }}
  scan(waterlineLines, 'waterline');
  scan(channelLines, 'channel');
  currentLine.forEach(function (p, pi) {{
    var d = distPx(latlng, L.latLng(p[0], p[1]));
    if (d < bestD) {{ bestD = d; best = {{current: true, pi: pi}}; }}
  }});
  if (!best) {{ return; }}
  if (best.current) {{ currentLine.splice(best.pi, 1); }}
  else {{
    var arr = best.type === 'waterline' ? waterlineLines : channelLines;
    arr[best.li].splice(best.pi, 1);
    if (arr[best.li].length === 0) {{ arr.splice(best.li, 1); }}
  }}
  redrawAll();
  sendState();
}}

map.on('click', function (e) {{
  var oe = e.originalEvent;
  if (oe && oe.ctrlKey) {{ insertPointIntoNearestLine(e.latlng); return; }}
  if (oe && oe.shiftKey) {{ deleteNearestPoint(e.latlng); return; }}
  if (!drawMode && !drawChannelMode) {{ return; }}
  var snapped = snapToEndpoint(e.latlng);
  currentLine.push([snapped.lat, snapped.lng]);
  redrawAll();
  sendState();
}});

map.on('contextmenu', function (e) {{
  if (currentLine.length > 0) {{ finishCurrentLine(); return; }}
  var hit = findLineNear(e.latlng);
  if (hit) {{ deleteLine(hit); }}
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


def build_terrain_html(terrain):
    """3D-вид дна (вкладка «Построение изобат» → «3D дно») — грид глубин
    (уже уменьшенный до разумного размера в _build_terrain_mesh) как меш в
    Three.js, загружаемом с CDN тем же способом, что и Leaflet в остальных
    картах. Окраска — та же jet-палитра и то же направление (мельче —
    красный, глубже — синий), что на 2D карте (build_map_html), но
    посчитанная в JS заново — это отдельная HTML-страница, свой JS-контекст,
    ничего нельзя переиспользовать между страницами напрямую. Ячейки без
    данных (null) не попадают ни в один треугольник — дыра в поверхности,
    как пустая область на 2D карте."""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>html,body{{height:100%;margin:0;background:#1e1e1e;overflow:hidden}}
#hint{{position:absolute;left:8px;bottom:6px;color:#aaa;font:12px sans-serif;
      pointer-events:none}}
#empty{{position:absolute;left:0;right:0;top:45%;text-align:center;color:#999;
       font:14px sans-serif;display:none}}</style>
</head><body>
<div id="hint">ЛКМ — вращение, колесо — зум, ПКМ — сдвиг</div>
<div id="empty">Нет данных для 3D-вида — постройте изобаты хотя бы с одной точкой глубины.</div>
<script>
var terrain = {json.dumps(terrain)};
var nx = terrain.nx, ny = terrain.ny;
var cellX = terrain.cellX || 1, cellY = terrain.cellY || 1;
var zmin = terrain.zmin, zmax = terrain.zmax;

var jetStops = [
  [0.0, [0, 0, 143]], [0.125, [0, 0, 255]], [0.375, [0, 255, 255]],
  [0.625, [255, 255, 0]], [0.875, [255, 0, 0]], [1.0, [128, 0, 0]]
];
function jetColorRGB(t) {{
  for (var i = 0; i < jetStops.length - 1; i++) {{
    var a = jetStops[i], b = jetStops[i + 1];
    if (t <= b[0] || i === jetStops.length - 2) {{
      var f = (t - a[0]) / (b[0] - a[0]);
      return [(a[1][0] + f * (b[1][0] - a[1][0])) / 255,
              (a[1][1] + f * (b[1][1] - a[1][1])) / 255,
              (a[1][2] + f * (b[1][2] - a[1][2])) / 255];
    }}
  }}
}}
function depthColorRGB(v) {{
  var t = Math.max(0, Math.min(1, (v - zmin) / ((zmax - zmin) || 1)));
  return jetColorRGB(1 - t);
}}

var scene = new THREE.Scene();
scene.background = new THREE.Color(0x1e1e1e);
var camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1e6);
var renderer = new THREE.WebGLRenderer({{antialias: true}});
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
var dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(1, 1.5, 1);
scene.add(dirLight);

var controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

var mesh = null, geometry = null, rawZ = [], zExaggeration = 3.0;

function buildMesh() {{
  if (nx < 2 || ny < 2) {{ document.getElementById('empty').style.display = 'block'; return; }}
  var centerX = (nx - 1) * cellX / 2, centerY = (ny - 1) * cellY / 2;
  var positions = new Float32Array(nx * ny * 3);
  var colors = new Float32Array(nx * ny * 3);
  rawZ = new Array(nx * ny);
  var any_valid = false;
  var p = 0;
  for (var j = 0; j < ny; j++) {{
    for (var i = 0; i < nx; i++) {{
      var v = terrain.z[j][i];
      positions[p * 3 + 0] = i * cellX - centerX;
      positions[p * 3 + 2] = j * cellY - centerY;
      if (v === null) {{
        positions[p * 3 + 1] = 0;
        rawZ[p] = null;
        colors[p * 3 + 0] = 0.25; colors[p * 3 + 1] = 0.25; colors[p * 3 + 2] = 0.25;
      }} else {{
        any_valid = true;
        positions[p * 3 + 1] = -v * zExaggeration;
        rawZ[p] = v;
        var c = depthColorRGB(v);
        colors[p * 3 + 0] = c[0]; colors[p * 3 + 1] = c[1]; colors[p * 3 + 2] = c[2];
      }}
      p++;
    }}
  }}
  if (!any_valid) {{ document.getElementById('empty').style.display = 'block'; return; }}
  var indices = [];
  for (var jj = 0; jj < ny - 1; jj++) {{
    for (var ii = 0; ii < nx - 1; ii++) {{
      if (terrain.z[jj][ii] === null || terrain.z[jj][ii + 1] === null ||
          terrain.z[jj + 1][ii] === null || terrain.z[jj + 1][ii + 1] === null) {{ continue; }}
      var a = jj * nx + ii, b = jj * nx + ii + 1, c2 = (jj + 1) * nx + ii, d = (jj + 1) * nx + ii + 1;
      indices.push(a, c2, b);
      indices.push(b, c2, d);
    }}
  }}
  geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  var material = new THREE.MeshLambertMaterial({{vertexColors: true, side: THREE.DoubleSide}});
  mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  var span = Math.max(nx * cellX, ny * cellY, 1);
  camera.position.set(span * 0.6, span * 0.5, span * 0.6);
  controls.target.set(0, 0, 0);
  controls.update();
}}
buildMesh();

function setExaggeration(v) {{
  zExaggeration = v;
  if (!geometry) {{ return; }}
  var pos = geometry.attributes.position;
  for (var k = 0; k < rawZ.length; k++) {{
    if (rawZ[k] !== null) {{ pos.setY(k, -rawZ[k] * zExaggeration); }}
  }}
  pos.needsUpdate = true;
  geometry.computeVertexNormals();
}}

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();

window.addEventListener('resize', function () {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body></html>"""


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
    со скроллом канваса (see EchogramTab).

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
    со скроллом канваса, как линейка — по вертикали. Разметка оси (время/
    расстояние) — как на таймлайне главной карты (build_map_html/drawTimeline):
    вертикальные линии на всю высоту графика + подписи снизу."""
    AXIS_H = 14
    HEIGHT = 60 + AXIS_H
    wheelZoom = Signal(float)  # множитель масштаба (>1 крупнее, <1 мельче) — см. apply_zoom_factor

    def __init__(self):
        super().__init__()
        self.setFixedHeight(self.HEIGHT)
        self.depth_m = []
        self.axis_vals = []
        self.axis_mode = "time"
        self.content_width = 0
        self.scroll_x = 0
        self.hover_col = None

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.wheelZoom.emit(factor)
        event.accept()

    def set_params(self, depth_m, axis_vals, axis_mode, content_width):
        self.depth_m = depth_m
        self.axis_vals = axis_vals
        self.axis_mode = axis_mode
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
        plot_h = h - self.AXIS_H
        dmin, dmax = min(self.depth_m), max(self.depth_m)
        if dmax <= dmin:
            dmax = dmin + 1e-6
        per_ping = self.content_width / n
        i0 = max(0, int(self.scroll_x / per_ping) - 1)
        i1 = min(n, int((self.scroll_x + w) / per_ping) + 2)

        if len(self.axis_vals) >= 2:
            a0, a1 = self.axis_vals[0], self.axis_vals[-1]
            ticks = max(2, min(8, w // 90))
            painter.setPen(QColor(255, 255, 255, 38))
            for k in range(ticks + 1):
                x = k / ticks * w
                painter.drawLine(int(x), 0, int(x), plot_h)
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            painter.setPen(QColor("#aaa"))
            for k in range(ticks + 1):
                frac = k / ticks
                x = frac * w
                label = format_axis_label(a0 + frac * (a1 - a0), self.axis_mode)
                if k == 0:
                    align = Qt.AlignmentFlag.AlignLeft
                elif k == ticks:
                    align = Qt.AlignmentFlag.AlignRight
                else:
                    align = Qt.AlignmentFlag.AlignHCenter
                text_rect = QRect(int(x) - 40, plot_h, 80, self.AXIS_H)
                painter.drawText(text_rect, int(align | Qt.AlignmentFlag.AlignVCenter), label)

        painter.setPen(QColor("#3388ff"))
        prev = None
        for i in range(i0, i1):
            x = i * per_ping - self.scroll_x
            t = (self.depth_m[i] - dmin) / (dmax - dmin)
            y = 4 + t * (plot_h - 8)
            if prev is not None:
                painter.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
            prev = (x, y)
        if self.hover_col is not None and 0 <= self.hover_col < n:
            x = self.hover_col * per_ping - self.scroll_x
            painter.setPen(QColor(255, 255, 0, 200))
            painter.drawLine(int(x), 0, int(x), plot_h)
            painter.setPen(QColor("white"))
            painter.drawText(min(w - 60, max(2, int(x) + 4)), 13,
                              f"{self.depth_m[self.hover_col]:.2f} м")


def build_echo_minimap_html(basemap, lat, lon):
    """Мини-карта в панели заметок вкладки «Эхограмма» — трек текущего файла
    и жёлтый маркер положения под курсором на эхограмме (setCursor/hideCursor,
    дёргается из Python при каждом hover_changed у EchogramCanvas)."""
    tile = BASEMAPS[basemap]
    center = [sum(lat) / len(lat), sum(lon) / len(lon)] if lat else [0, 0]
    pts = list(zip(lat, lon))
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}}</style>
</head><body>
<div id="map"></div>
<script>
var map = L.map('map', {{preferCanvas: true, attributionControl: false, zoomControl: false}})
  .setView([{center[0]}, {center[1]}], 14);
L.tileLayer('{tile["url"]}', {{maxZoom: {tile["max_zoom"]}}}).addTo(map);
var pts = {json.dumps(pts)};
if (pts.length > 1) {{
  var line = L.polyline(pts, {{color: '#ff3030', weight: 2}}).addTo(map);
  map.fitBounds(line.getBounds());
}} else if (pts.length === 1) {{
  map.setView(pts[0], 16);
}}
var cursor = L.circleMarker([0, 0], {{radius: 6, color: '#fff', weight: 2,
                                      fillColor: '#ffd400', fillOpacity: 1}});
function setCursor(lat, lon) {{
  cursor.setLatLng([lat, lon]);
  if (!cursor._map) {{ cursor.addTo(map); }}
}}
function hideCursor() {{
  if (cursor._map) {{ map.removeLayer(cursor); }}
}}
</script>
</body></html>"""


class EchogramCanvas(QWidget):
    AXIS_H = 22
    zoom_changed = Signal()
    hover_changed = Signal(object)  # индекс пинга под курсором, либо None
    markRequested = Signal(int, float)  # ping_idx, row (байт по глубине) — клик в режиме "mark"
    rulerChanged = Signal(object)       # dict измерения (см. ruler_measurement) либо None

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
        self.mode = "normal"    # normal | mark | ruler
        self.marks = []         # [{"ping_idx", "row", "axis_value", "depth_m", "text"}, ...]
        self.ruler_points = []  # до двух (ping_idx, row) в координатах данных
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
        self._draw_marks(painter, w, h)
        self._draw_ruler(painter, w, h)
        if self.hover_pos is not None:
            self._draw_crosshair(painter, w, h)

    def _data_to_pixel(self, col, row):
        rows, cols = self.arr.shape
        w = int(cols * self.zoom)
        h = int(rows * self.zoom)
        x = (col / len(self.axis_vals) * w) if self.axis_vals else (col / cols * w)
        y = row / rows * h
        return x, y

    def _pixel_to_data(self, x, y):
        if self.arr is None:
            return None
        rows, cols = self.arr.shape
        w = int(cols * self.zoom)
        h = int(rows * self.zoom)
        if w <= 0 or h <= 0:
            return None
        col = int(x / w * len(self.axis_vals)) if self.axis_vals else int(x / w * cols)
        col = max(0, min(cols - 1, col))
        row = max(0.0, min(float(rows), y / h * rows))
        return col, row

    def ruler_measurement(self):
        if len(self.ruler_points) < 2:
            return None
        (c1, r1), (c2, r2) = self.ruler_points
        d_rows = abs(r2 - r1)
        depth_m = (d_rows / self.arr.shape[0] * self.range_m) if self.range_m is not None else None
        axis_delta = None
        if self.axis_vals and 0 <= c1 < len(self.axis_vals) and 0 <= c2 < len(self.axis_vals):
            axis_delta = abs(float(self.axis_vals[c2]) - float(self.axis_vals[c1]))
        return dict(depth_m=depth_m, axis_delta=axis_delta, axis_mode=self.axis_mode,
                    d_pings=abs(c2 - c1))

    def _draw_marks(self, painter, w, h):
        for i, m in enumerate(self.marks):
            x, y = self._data_to_pixel(m["ping_idx"], m["row"])
            if not (0 <= x <= w and 0 <= y <= h):
                continue
            painter.setPen(QColor("black"))
            painter.setBrush(QColor("#ffcc00"))
            painter.drawEllipse(int(x) - 5, int(y) - 5, 10, 10)
            label = (m.get("text") or "").splitlines()[0][:24] or f"#{i + 1}"
            painter.setPen(QColor("#ffcc00"))
            painter.drawText(int(x) + 8, int(y) + 4, label)

    def _draw_ruler(self, painter, w, h):
        if not self.ruler_points:
            return
        pts_px = [self._data_to_pixel(c, r) for c, r in self.ruler_points]
        painter.setPen(QColor("#00e5ff"))
        painter.setBrush(QColor("#00e5ff"))
        for x, y in pts_px:
            painter.drawEllipse(int(x) - 4, int(y) - 4, 8, 8)
        if len(pts_px) < 2:
            return
        (x1, y1), (x2, y2) = pts_px
        painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        meas = self.ruler_measurement()
        if not meas:
            return
        parts = []
        if meas["depth_m"] is not None:
            parts.append(f"Δглубина {meas['depth_m']:.2f} м")
        if meas["axis_delta"] is not None:
            parts.append(format_axis_label(meas["axis_delta"], meas["axis_mode"]))
        label = "  ".join(parts)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        fm = painter.fontMetrics()
        box_w = fm.horizontalAdvance(label) + 8
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 229, 255, 220))
        painter.drawRect(int(mx) - box_w // 2, int(my) - 15, box_w, 14)
        painter.setPen(QColor("black"))
        painter.drawText(int(mx) - box_w // 2 + 4, int(my) - 4, label)

    def mousePressEvent(self, event):
        if self.arr is None or self.mode == "normal":
            return
        pos = event.position() if hasattr(event, "position") else event.pos()
        data = self._pixel_to_data(pos.x(), pos.y())
        if data is None:
            return
        col, row = data
        if self.mode == "mark":
            self.markRequested.emit(col, row)
        elif self.mode == "ruler":
            if len(self.ruler_points) >= 2:
                self.ruler_points = []
            self.ruler_points.append((col, row))
            self.rulerChanged.emit(self.ruler_measurement())
            self.update()

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

    def apply_zoom_factor(self, factor):
        """Меняет масштаб на factor (>1 — крупнее, <1 — мельче) — общий код для
        колеса мыши над самой эхограммой и над таймлайном под ней (см.
        EchoTimelineWidget.wheelZoom)."""
        if self.image is None:
            return
        self.zoom = max(0.1, min(20.0, self.zoom * factor))
        self._rebuild_image()
        self.updateGeometry()
        self.resize(self.sizeHint())
        self.update()
        self.zoom_changed.emit()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.apply_zoom_factor(factor)
        event.accept()


class EchogramTab(QWidget):
    """Просмотр эхограммы (водопад сонара) — раньше отдельное модальное окно
    (`EchogramViewDialog`), теперь вкладка: обновляется, только когда меняется
    путь к файлу эхограммы в шапке окна (само чтение/декодирование водопада не
    дешёвое, поэтому не перезагружаем на каждое переключение вкладки, как
    IsobathsTab — см. refresh())."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._loaded_path = None
        self.echo_lat = []
        self.echo_lon = []

        self.status_label = QLabel(
            "Выберите один файл эхограммы (.sl2) в шапке окна, чтобы посмотреть "
            "её здесь (мультиэхограмма не поддерживается).")
        self.canvas = EchogramCanvas()
        self.canvas.zoom_changed.connect(self.sync_side_widgets)
        self.canvas.markRequested.connect(self.on_mark_requested)
        self.canvas.rulerChanged.connect(self.on_ruler_changed)
        self.ruler_widget = DepthRulerWidget(self.canvas)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)
        self.scroll.verticalScrollBar().valueChanged.connect(self.ruler_widget.set_scroll_offset)

        self.mark_mode_btn = QPushButton("Режим заметок")
        self.mark_mode_btn.setCheckable(True)
        self.mark_mode_btn.setToolTip(
            "Включите и кликните по эхограмме, чтобы поставить заметку с текстом "
            "в этом месте.")
        self.mark_mode_btn.toggled.connect(self.on_mark_mode_toggled)
        self.ruler_mode_btn = QPushButton("Линейка")
        self.ruler_mode_btn.setCheckable(True)
        self.ruler_mode_btn.setToolTip(
            "Включите и кликните по двум точкам на эхограмме — покажет "
            "расстояние между ними по глубине и по времени/дистанции.")
        self.ruler_mode_btn.toggled.connect(self.on_ruler_mode_toggled)
        clear_ruler_btn = QPushButton("Сбросить линейку")
        clear_ruler_btn.clicked.connect(self.clear_ruler)
        save_marks_btn = QPushButton("Сохранить заметки")
        save_marks_btn.clicked.connect(self.save_marks)
        load_marks_btn = QPushButton("Загрузить заметки")
        load_marks_btn.clicked.connect(self.load_marks)

        toolbar_row = QHBoxLayout()
        toolbar_row.addWidget(self.mark_mode_btn)
        toolbar_row.addWidget(self.ruler_mode_btn)
        toolbar_row.addWidget(clear_ruler_btn)
        toolbar_row.addStretch(1)
        toolbar_row.addWidget(save_marks_btn)
        toolbar_row.addWidget(load_marks_btn)

        self.ruler_label = QLabel("")

        self.marks_list = QListWidget()
        self.marks_list.currentRowChanged.connect(self.on_mark_selected)
        self.mark_text_edit = QTextEdit()
        self.mark_text_edit.textChanged.connect(self.on_mark_text_changed)
        remove_mark_btn = QPushButton("Удалить заметку")
        remove_mark_btn.clicked.connect(self.remove_selected_mark)

        self.mini_map = new_map_view()
        self.mini_map.setFixedHeight(200)
        load_html(self.mini_map, build_echo_minimap_html(
            self.main_window.basemap_combo.currentText(), [], []), "echo_minimap")

        marks_layout = QVBoxLayout()
        marks_layout.addWidget(QLabel("Заметки:"))
        marks_layout.addWidget(self.marks_list, 1)
        marks_layout.addWidget(QLabel("Текст заметки:"))
        marks_layout.addWidget(self.mark_text_edit, 1)
        marks_layout.addWidget(remove_mark_btn)
        marks_layout.addWidget(self.mini_map)
        marks_widget = QWidget()
        marks_widget.setLayout(marks_layout)
        marks_widget.setFixedWidth(260)

        content_row = QHBoxLayout()
        content_row.addWidget(self.ruler_widget)
        content_row.addWidget(self.scroll, 1)
        content_row.addWidget(marks_widget)

        self.echo_timeline = EchoTimelineWidget()
        self.canvas.hover_changed.connect(self.echo_timeline.set_hover_col)
        self.canvas.hover_changed.connect(self.on_hover_position)
        self.echo_timeline.wheelZoom.connect(self.canvas.apply_zoom_factor)
        self.scroll.horizontalScrollBar().valueChanged.connect(self.echo_timeline.set_scroll_offset)
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
        layout.addLayout(toolbar_row)
        layout.addWidget(self.ruler_label)
        layout.addLayout(content_row, 1)
        layout.addLayout(timeline_row)
        layout.addLayout(contrast_row)
        self.setLayout(layout)

    def refresh(self):
        path = self.main_window.sl2_edit.text()
        if not path or ";" in path:
            self._loaded_path = None
            self.status_label.setText(
                "Выберите один файл эхограммы (.sl2) в шапке окна, чтобы "
                "посмотреть её здесь (мультиэхограмма не поддерживается).")
            return
        if path == self._loaded_path:
            return
        self._loaded_path = path
        self.canvas.marks = []
        self.canvas.ruler_points = []
        self._refresh_marks_list()
        self.ruler_label.setText("")
        self.status_label.setText("Чтение файла…")
        run_async(lambda: read_echogram_waterfall(path),
                  on_finished=self.on_loaded, on_error=self.on_error)

    def sync_side_widgets(self):
        if self.canvas.image is None or self.canvas.arr is None:
            return
        rows, cols = self.canvas.arr.shape
        content_height = int(rows * self.canvas.zoom)
        self.ruler_widget.set_params(rows, content_height, self.canvas.range_m)
        content_width = int(cols * self.canvas.zoom)
        self.echo_timeline.set_params(self.canvas.depth_m, self.canvas.axis_vals,
                                       self.canvas.axis_mode, content_width)

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
        name = os.path.basename(self._loaded_path) if self._loaded_path else ""
        self.status_label.setText(f"{name}: {arr.shape[1]} пингов, {arr.shape[0]} байт по "
                                   f"глубине (колесо мыши — масштаб){note}")
        axis_mode = self.main_window.settings.value("timeline_axis", "time")
        axis_vals = data["dist"] if axis_mode == "distance" else data["t_rel"]
        self.canvas.set_data(arr, valid, data["depth_m"], axis_vals, axis_mode, cal_range)
        self.sync_side_widgets()
        self.echo_lat = data.get("lat", [])
        self.echo_lon = data.get("lon", [])
        load_html(self.mini_map, build_echo_minimap_html(
            self.main_window.basemap_combo.currentText(), self.echo_lat, self.echo_lon),
            "echo_minimap")

    def on_error(self, message):
        self.status_label.setText(f"Ошибка чтения: {message}")

    def on_hover_position(self, col):
        if col is None or not self.echo_lat or col >= len(self.echo_lat):
            self.mini_map.page().runJavaScript("hideCursor();")
            return
        self.mini_map.page().runJavaScript(
            f"setCursor({self.echo_lat[col]}, {self.echo_lon[col]});")

    def on_mark_mode_toggled(self, checked):
        if checked:
            self.ruler_mode_btn.setChecked(False)
            self.canvas.mode = "mark"
        elif self.canvas.mode == "mark":
            self.canvas.mode = "normal"

    def on_ruler_mode_toggled(self, checked):
        if checked:
            self.mark_mode_btn.setChecked(False)
            self.canvas.mode = "ruler"
        elif self.canvas.mode == "ruler":
            self.canvas.mode = "normal"

    def clear_ruler(self):
        self.canvas.ruler_points = []
        self.ruler_label.setText("")
        self.canvas.update()

    def on_ruler_changed(self, measurement):
        if not measurement:
            self.ruler_label.setText("")
            return
        parts = [f"пингов между точками: {measurement['d_pings']}"]
        if measurement["depth_m"] is not None:
            parts.append(f"Δ глубина: {measurement['depth_m']:.2f} м")
        if measurement["axis_delta"] is not None:
            axis_name = "расстояние" if measurement["axis_mode"] == "distance" else "время"
            parts.append(f"Δ {axis_name}: "
                         f"{format_axis_label(measurement['axis_delta'], measurement['axis_mode'])}")
        self.ruler_label.setText("Линейка: " + ", ".join(parts))

    def on_mark_requested(self, ping_idx, row):
        text, ok = QInputDialog.getMultiLineText(self, "Заметка", "Текст заметки:")
        if not ok:
            return
        axis_value = None
        if self.canvas.axis_vals and 0 <= ping_idx < len(self.canvas.axis_vals):
            axis_value = float(self.canvas.axis_vals[ping_idx])
        depth_m = None
        if self.canvas.range_m is not None and self.canvas.arr is not None:
            depth_m = float(row / self.canvas.arr.shape[0] * self.canvas.range_m)
        mark = dict(ping_idx=int(ping_idx), row=float(row), axis_value=axis_value,
                   depth_m=depth_m, text=text)
        self.canvas.marks.append(mark)
        self.canvas.update()
        self._refresh_marks_list()
        self.marks_list.setCurrentRow(len(self.canvas.marks) - 1)

    def _mark_label(self, mark):
        axis_mode = self.canvas.axis_mode
        if mark.get("axis_value") is not None:
            pos = format_axis_label(mark["axis_value"], axis_mode)
        else:
            pos = f"пинг {mark['ping_idx']}"
        depth = f", {mark['depth_m']:.1f} м" if mark.get("depth_m") is not None else ""
        text = (mark.get("text") or "").splitlines()
        text = text[0][:40] if text else "(без текста)"
        return f"{pos}{depth} — {text}"

    def _refresh_marks_list(self):
        self.marks_list.blockSignals(True)
        self.marks_list.clear()
        for m in self.canvas.marks:
            self.marks_list.addItem(self._mark_label(m))
        self.marks_list.blockSignals(False)

    def on_mark_selected(self, row):
        self.mark_text_edit.blockSignals(True)
        self.mark_text_edit.setPlainText(
            self.canvas.marks[row].get("text", "") if 0 <= row < len(self.canvas.marks) else "")
        self.mark_text_edit.blockSignals(False)

    def on_mark_text_changed(self):
        row = self.marks_list.currentRow()
        if not (0 <= row < len(self.canvas.marks)):
            return
        self.canvas.marks[row]["text"] = self.mark_text_edit.toPlainText()
        self.marks_list.blockSignals(True)
        self.marks_list.item(row).setText(self._mark_label(self.canvas.marks[row]))
        self.marks_list.blockSignals(False)
        self.canvas.update()

    def remove_selected_mark(self):
        row = self.marks_list.currentRow()
        if 0 <= row < len(self.canvas.marks):
            del self.canvas.marks[row]
            self._refresh_marks_list()
            self.canvas.update()

    def save_marks(self):
        if not self.canvas.marks:
            QMessageBox.information(self, "Заметки", "Нет заметок для сохранения.")
            return
        base = os.path.splitext(self._loaded_path)[0] if self._loaded_path else "echogram"
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить заметки", base + "_marks.json", "JSON (*.json)")
        if not path:
            return
        payload = dict(sl2_path=self._loaded_path, marks=self.canvas.marks)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except OSError as e:
            QMessageBox.warning(self, "Заметки", f"Не удалось сохранить:\n{e}")
            return
        self.status_label.setText(f"Заметки сохранены: {path}")

    def load_marks(self):
        start_dir = os.path.dirname(self._loaded_path) if self._loaded_path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить заметки", start_dir, "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Заметки", f"Не удалось прочитать файл:\n{e}")
            return
        marks = payload.get("marks", [])
        if self.canvas.arr is not None:
            n = self.canvas.arr.shape[1]
            marks = [m for m in marks if 0 <= m.get("ping_idx", -1) < n]
        self.canvas.marks = marks
        self.canvas.update()
        self._refresh_marks_list()
        self.status_label.setText(f"Загружено заметок: {len(marks)} из {path}")


class IsobathsLinesBridge(QObject):
    """Мост QWebChannel для карты на вкладке изобат — JS сам ведёт состояние
    линий уреза/русла (добавление/вставка/удаление точек и целых линий,
    прилипание концов, курсор) и очистки точек трека (клик — исключить/
    вернуть) и после каждого изменения сообщает сюда; Python просто
    запоминает для build_isobaths (тот же поток, без QThread — сюда правило
    про сигналы между потоками и QWebEngineView не относится)."""

    def __init__(self, on_change, on_point_toggled):
        super().__init__()
        self.on_change = on_change
        self.on_point_toggled = on_point_toggled

    @Slot(str)
    def linesChanged(self, payload_json):
        self.on_change(json.loads(payload_json))

    @Slot(int)
    def pointToggled(self, index):
        self.on_point_toggled(index)


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


class HistogramWidget(QWidget):
    """Гистограмма распределения индекса твёрдости дна (0..1) — см. SedimentsTab."""
    N_BINS = 20

    def __init__(self):
        super().__init__()
        self.values = []
        self.setMinimumHeight(160)

    def set_values(self, values):
        self.values = [v for v in values if v is not None and np.isfinite(v)]
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))
        if not self.values:
            painter.setPen(QColor("#888888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Нет данных")
            return
        counts, _edges = np.histogram(self.values, bins=self.N_BINS, range=(0.0, 1.0))
        w, h = self.width(), self.height()
        margin_b = 16
        max_count = counts.max() if counts.max() > 0 else 1
        bar_w = w / self.N_BINS
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#3388ff"))
        for i, c in enumerate(counts):
            bar_h = (c / max_count) * (h - margin_b - 4)
            x = i * bar_w
            y = h - margin_b - bar_h
            painter.drawRect(int(x) + 1, int(y), max(1, int(bar_w) - 2), int(bar_h))
        painter.setPen(QColor("#cccccc"))
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = frac * w
            painter.drawText(int(x) - 10, h - 2, f"{frac:.2f}")


class SedimentsTab(QWidget):
    """Аналитика по твёрдости дна (экспериментальный индекс, не откалиброван —
    см. README «Твёрдость дна») — карта с той же окраской, что «Раскраска:
    Твёрдость дна» на вкладке «Карта», плюс гистограмма распределения и
    статистика. Обновляется при каждом переключении на вкладку — дёшево,
    просто перечитывает main_window.points без пересчёта."""

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        note = QLabel(
            "Индекс твёрдости дна — экспериментальный (не откалиброван, не привязан "
            "к реальным типам грунта) — относительный показатель для сравнения точек "
            "в пределах одной записи. Подробности — README.md → «Твёрдость дна».")
        note.setWordWrap(True)
        note.setStyleSheet("color: #999;")

        self.map_view = new_map_view()

        self.stats_label = QLabel("Нет данных — загрузите эхограмму.")
        self.stats_label.setWordWrap(True)
        self.histogram = HistogramWidget()

        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Распределение индекса (0 — мягкое дно, 1 — твёрдое):"))
        right_col.addWidget(self.histogram)
        right_col.addWidget(self.stats_label)
        right_col.addStretch(1)
        right_widget = QWidget()
        right_widget.setLayout(right_col)
        right_widget.setFixedWidth(320)

        content_row = QHBoxLayout()
        content_row.addWidget(self.map_view, 1)
        content_row.addWidget(right_widget)

        layout = QVBoxLayout()
        layout.addWidget(note)
        layout.addLayout(content_row, 1)
        self.setLayout(layout)

        self.refresh()

    def refresh(self):
        mw = self.main_window
        manual = mw.settings.value("hardness_mode", "auto") == "manual"
        vmin = float(mw.settings.value("hardness_min", 0.0)) if manual else 0.0
        vmax = float(mw.settings.value("hardness_max", 1.0)) if manual else 1.0
        steps = int(mw.settings.value("hardness_steps", 32))
        points = mw.points
        hardness = (points or {}).get("hardness", [])
        if not points or not hardness:
            self.stats_label.setText("Нет данных — загрузите эхограмму.")
            self.histogram.set_values([])
            load_html(self.map_view, build_map_html(
                mw.basemap_combo.currentText(), dict(lat=[], lon=[], value=[]),
                True, vmin=vmin, vmax=vmax, legend_title="Твёрдость дна"), "sediments")
            return
        arr_h = np.asarray(hardness, dtype=float)
        arr_d = np.asarray(points.get("value", []), dtype=float)
        finite = np.isfinite(arr_h)
        arr_h_f = arr_h[finite]
        self.histogram.set_values(arr_h_f.tolist())
        if len(arr_h_f):
            corr_line = ""
            both = finite & np.isfinite(arr_d)
            if both.sum() > 2:
                r = float(np.corrcoef(arr_h[both], arr_d[both])[0, 1])
                corr_line = f"\nКорреляция с глубиной: {r:+.2f}"
            self.stats_label.setText(
                f"Точек: {len(arr_h_f)}\n"
                f"Среднее: {arr_h_f.mean():.3f}\n"
                f"Медиана: {float(np.median(arr_h_f)):.3f}\n"
                f"СКО: {arr_h_f.std():.3f}\n"
                f"Мин/макс: {arr_h_f.min():.3f} / {arr_h_f.max():.3f}"
                f"{corr_line}")
        else:
            self.stats_label.setText("Нет валидных значений твёрдости дна.")
        map_points = dict(points, value=hardness)
        html = build_map_html(mw.basemap_combo.currentText(), map_points, True,
                              vmin=vmin, vmax=vmax, steps=steps,
                              legend_title="Твёрдость дна", tooltip_suffix="", show_endpoints=False)
        load_html(self.map_view, html, "sediments")


class IsobathsTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.waterline_lines = []
        self.channel_lines = []
        self.last_result = None
        self.line_color = QColor("#000000")
        self.excluded_indices = set()

        self.bridge = IsobathsLinesBridge(self.on_lines_changed, self.on_point_toggled)
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

        points_group = QGroupBox("Точки")
        self.clean_btn = QPushButton("Удалить точки")
        self.clean_btn.setCheckable(True)
        self.clean_btn.setToolTip(
            "Клик по точке трека на карте исключает её из расчёта изобат "
            "(клик по ней ещё раз — вернуть). Для явных выбросов промера "
            "(шум, всплытие датчика), которые портят интерполяцию.")
        self.clean_btn.toggled.connect(self.on_clean_toggled)
        self.points_label = QLabel("Исключено точек: 0")
        restore_points_btn = QPushButton("Восстановить все")
        restore_points_btn.clicked.connect(self.restore_all_points)
        points_layout = QVBoxLayout()
        points_layout.addWidget(self.clean_btn)
        points_layout.addWidget(self.points_label)
        points_layout.addWidget(restore_points_btn)
        points_group.setLayout(points_layout)

        waterline_group = QGroupBox("Урез")
        load_waterline_btn = QPushButton("Загрузить урез")
        load_waterline_btn.clicked.connect(self.load_waterline_from_file)
        load_waterline_row = QHBoxLayout()
        load_waterline_row.addWidget(load_waterline_btn, 1)
        load_waterline_row.addWidget(help_icon(WATERLINE_FILE_HELP))

        self.draw_btn = QPushButton("Нарисовать\nурез")
        self.draw_btn.setCheckable(True)
        self.draw_btn.toggled.connect(self.on_draw_toggled)
        self.draw_channel_btn = QPushButton("Нарисовать\nрусло")
        self.draw_channel_btn.setCheckable(True)
        self.draw_channel_btn.toggled.connect(self.on_draw_channel_toggled)
        draw_tip = ("Клик — точка (концы линий притягиваются друг к другу). ПКМ — "
                     "закончить линию, а по уже нарисованной линии — удалить её. "
                     "Ctrl+клик — вставить точку в ближайшую линию. Shift+клик — "
                     "удалить ближайшую точку. Линий каждого типа может быть несколько.")
        self.draw_btn.setToolTip(draw_tip)
        self.draw_channel_btn.setToolTip(draw_tip)
        draw_row = QHBoxLayout()
        draw_row.addWidget(self.draw_btn)
        draw_row.addWidget(self.draw_channel_btn)

        self.corner_smooth_check = QCheckBox("Сглаживание")
        self.corner_smooth_check.setToolTip(
            "Сглаживает острые углы нарисованных вручную линий уреза и русла "
            "(начальная и конечная точки каждой линии остаются на месте) — "
            "видно сразу на карте, действует и при нажатии «Построить».")
        self.corner_smooth_check.toggled.connect(self.on_corner_smooth_toggled)
        self.hide_waterline_check = QCheckBox("Скрыть")
        self.hide_waterline_check.toggled.connect(self.on_hide_waterline_toggled)
        options_row = QHBoxLayout()
        options_row.addWidget(self.corner_smooth_check)
        options_row.addWidget(self.hide_waterline_check)

        waterline_layout = QVBoxLayout()
        waterline_layout.addLayout(load_waterline_row)
        waterline_layout.addLayout(draw_row)
        waterline_layout.addLayout(options_row)
        waterline_group.setLayout(waterline_layout)

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
        self.label_freq_spin = QSlider(Qt.Orientation.Horizontal)
        self.label_freq_spin.setRange(5, 500)
        self.label_freq_spin.setValue(40)
        self.label_freq_spin.valueChanged.connect(self.apply_style)
        label_freq_row = QHBoxLayout()
        label_freq_row.addWidget(QLabel("Частота подписей:"))
        label_freq_row.addWidget(self.label_freq_spin)
        self.label_nobg_check = QCheckBox("Только цифры, без фона")
        self.label_nobg_check.setChecked(False)
        self.label_nobg_check.toggled.connect(self.apply_style)
        label_layout = QVBoxLayout()
        label_layout.addWidget(self.show_labels_check)
        label_layout.addLayout(label_size_row)
        label_layout.addLayout(label_freq_row)
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
        self.fill_steps_spin.setRange(0, 30)
        self.fill_steps_spin.setValue(8)
        self.fill_steps_spin.setToolTip(
            "Число ступеней перехода цвета заливки от минимальной глубины к "
            "максимальной. 0 — перехода нет, вся заливка одним цветом. "
            "10 — плавный переход в 10 цветов.")
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
        settings_layout.addWidget(points_group)
        settings_layout.addWidget(waterline_group)
        settings_layout.addWidget(line_group)
        settings_layout.addWidget(label_group)
        settings_layout.addWidget(fill_group)
        settings_layout.addStretch(1)
        settings_widget = QWidget()
        settings_widget.setLayout(settings_layout)
        settings_widget.setFixedWidth(240)

        self.view_3d = new_map_view()
        load_html(self.view_3d, build_terrain_html(dict(nx=0, ny=0, x0=0, y0=0, cellX=1, cellY=1,
                                                          z=[], zmin=0, zmax=1)), "terrain3d")
        self.map_stack = QStackedWidget()
        self.map_stack.addWidget(self.map_view)
        self.map_stack.addWidget(self.view_3d)

        self.view2d_btn = QPushButton("Карта")
        self.view2d_btn.setCheckable(True)
        self.view2d_btn.setChecked(True)
        self.view2d_btn.toggled.connect(self.on_view2d_toggled)
        self.view3d_btn = QPushButton("3D дно")
        self.view3d_btn.setCheckable(True)
        self.view3d_btn.toggled.connect(self.on_view3d_toggled)
        self.exaggeration_spin = QDoubleSpinBox()
        self.exaggeration_spin.setRange(1.0, 30.0)
        self.exaggeration_spin.setValue(3.0)
        self.exaggeration_spin.setSuffix("×")
        self.exaggeration_spin.setToolTip("Вертикальное преувеличение рельефа дна в 3D — "
                                           "глубина обычно мала по сравнению с площадью акватории, "
                                           "без преувеличения рельеф почти не виден.")
        self.exaggeration_spin.valueChanged.connect(self.on_exaggeration_changed)
        view_row = QHBoxLayout()
        view_row.addWidget(self.view2d_btn)
        view_row.addWidget(self.view3d_btn)
        view_row.addSpacing(12)
        view_row.addWidget(QLabel("Преувеличение рельефа:"))
        view_row.addWidget(self.exaggeration_spin)
        view_row.addStretch(1)

        map_col = QVBoxLayout()
        map_col.addLayout(view_row)
        map_col.addWidget(self.map_stack, 1)

        content_row = QHBoxLayout()
        content_row.addWidget(settings_widget)
        content_row.addLayout(map_col, 1)

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

    def refresh_track(self):
        """Перечитывает трек/подложку с главного окна — вызывается при каждом
        переключении на эту вкладку (раньше карта строилась один раз при
        открытии диалога; как постоянная вкладка, она должна видеть свежие
        данные, если пользователь загрузил другой файл, не заходя сюда).
        Уже построенные изобаты/урез не трогает — они просто не будут видны
        поверх новой карты, пока не нажать «Построить» заново (как и раньше
        было при повторном открытии диалога)."""
        points = self.main_window.points or {}
        lat = points.get("lat", [])
        if len(lat) != getattr(self, "_last_points_len", None):
            # Число точек изменилось — скорее всего, загружена другая запись:
            # старые индексы исключённых точек больше ничего не значат.
            self.excluded_indices = set()
            self.points_label.setText("Исключено точек: 0")
        self._last_points_len = len(lat)
        html = build_isobaths_map_html(self.main_window.basemap_combo.currentText(),
                                        lat, points.get("lon", []),
                                        excluded_indices=self.excluded_indices)
        load_html(self.map_view, html, "isobaths")

    def on_draw_toggled(self, checked):
        if checked:
            self.draw_channel_btn.setChecked(False)
            self.clean_btn.setChecked(False)
        self.draw_btn.setText("Клик по\nкарте…" if checked else "Нарисовать\nурез")
        self.map_view.page().runJavaScript(f"setDrawMode({'true' if checked else 'false'});")

    def on_draw_channel_toggled(self, checked):
        if checked:
            self.draw_btn.setChecked(False)
            self.clean_btn.setChecked(False)
        self.draw_channel_btn.setText("Клик по\nкарте…" if checked else "Нарисовать\nрусло")
        self.map_view.page().runJavaScript(f"setDrawChannelMode({'true' if checked else 'false'});")

    def on_clean_toggled(self, checked):
        if checked:
            self.draw_btn.setChecked(False)
            self.draw_channel_btn.setChecked(False)
        self.clean_btn.setText("Клик по\nкарте…" if checked else "Удалить точки")
        self.map_view.page().runJavaScript(f"setCleanMode({'true' if checked else 'false'});")

    def on_point_toggled(self, index):
        if index in self.excluded_indices:
            self.excluded_indices.discard(index)
        else:
            self.excluded_indices.add(index)
        self.points_label.setText(f"Исключено точек: {len(self.excluded_indices)}")

    def restore_all_points(self):
        self.excluded_indices = set()
        self.points_label.setText("Исключено точек: 0")
        self.map_view.page().runJavaScript("restoreAllPoints();")

    def on_lines_changed(self, payload):
        self.waterline_lines = [[(p[0], p[1]) for p in line] for line in payload.get("waterline", [])]
        self.channel_lines = [[(p[0], p[1]) for p in line] for line in payload.get("channel", [])]

    def on_hide_waterline_toggled(self, checked):
        self.map_view.page().runJavaScript(f"setWaterlineVisible({'false' if checked else 'true'});")

    def on_corner_smooth_toggled(self, checked):
        self.map_view.page().runJavaScript(f"setSmoothPreview({'true' if checked else 'false'});")

    def on_hide_track_toggled(self, checked):
        self.hide_track_btn.setText("Показать трек" if checked else "Скрыть трек")
        self.map_view.page().runJavaScript(f"setTrackVisible({'false' if checked else 'true'});")

    def on_view2d_toggled(self, checked):
        if checked:
            self.view3d_btn.setChecked(False)
            self.map_stack.setCurrentWidget(self.map_view)
        elif not self.view3d_btn.isChecked():
            self.view2d_btn.setChecked(True)

    def on_view3d_toggled(self, checked):
        if checked:
            self.view2d_btn.setChecked(False)
            self.map_stack.setCurrentWidget(self.view_3d)
        elif not self.view2d_btn.isChecked():
            self.view3d_btn.setChecked(True)

    def on_exaggeration_changed(self, value):
        self.view_3d.page().runJavaScript(f"setExaggeration({value});")

    def load_waterline_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить урез воды", "",
            "Текст/CSV/KML (*.csv *.txt *.kml);;Все файлы (*)")
        if not path:
            return
        try:
            if path.lower().endswith(".kml"):
                urez_lines, channel_lines = read_waterline_kml(path)
            else:
                urez_lines, channel_lines = [read_waterline_points(path)], []
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "Урез воды", f"Не удалось прочитать файл:\n{e}")
            return
        self.waterline_lines.extend(urez_lines)
        self.channel_lines.extend(channel_lines)
        payload = json.dumps(dict(
            waterline=[[[lat, lon] for lat, lon in line] for line in urez_lines],
            channel=[[[lat, lon] for lat, lon in line] for line in channel_lines]))
        self.map_view.page().runJavaScript(f"addLines({payload});")

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
        if self.excluded_indices:
            keep = [i for i in range(len(lat)) if i not in self.excluded_indices]
            lat = [lat[i] for i in keep]
            lon = [lon[i] for i in keep]
            depth = [depth[i] for i in keep]
        if len(lat) < 10:
            QMessageBox.information(self, "Изобаты",
                                     "Недостаточно точек — сначала загрузите эхограмму.")
            return
        waterline_lines = [list(line) for line in self.waterline_lines]
        channel_lines = [list(line) for line in self.channel_lines]
        if self.corner_smooth_check.isChecked():
            waterline_lines = [smooth_polyline_corners(line) for line in waterline_lines]
            channel_lines = [smooth_polyline_corners(line) for line in channel_lines]
        self.status_label.setText("Строю изобаты…")
        run_async(lambda: compute_isobaths(lat, lon, depth, waterline_lines,
                                            self.cell_spin.value(), self.interval_spin.value(),
                                            self.fill_steps_spin.value(), self.smooth_spin.value(),
                                            channel_lines=channel_lines),
                  on_finished=self.on_built, on_error=self.on_build_error)

    def on_built(self, result):
        self.last_result = result
        self.status_label.setText(f"Построено изобат: {len(result['lines'])}, "
                                   f"диапазонов заливки: {len(result['bands'])}")
        self.apply_style()
        self.on_show_lines_toggled(self.show_lines_check.isChecked())
        self.on_show_fill_toggled(self.show_fill_check.isChecked())
        self.on_show_labels_toggled(self.show_labels_check.isChecked())
        self.rebuild_3d_view()

    def rebuild_3d_view(self):
        terrain = (self.last_result or {}).get("terrain")
        if not terrain:
            return
        html = build_terrain_html(terrain)
        load_html(self.view_3d, html, "terrain3d")
        # Свежая страница — свой JS-контекст со значением по умолчанию (3×),
        # применяем текущее выбранное преувеличение сразу после загрузки.
        QTimer.singleShot(400, lambda: self.on_exaggeration_changed(self.exaggeration_spin.value()))

    def delete_isobaths(self):
        self.last_result = None
        self.status_label.setText("Изобаты удалены")
        self.map_view.page().runJavaScript("clearIsobaths(); setFillLegendVisible(false);")

    def save_as_image(self):
        sl2_path = self.main_window.sl2_edit.text()
        default = ""
        if sl2_path and ";" not in sl2_path:
            default = os.path.splitext(sl2_path)[0] + "_isobaths.png"
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить изображение", default, "PNG (*.png)")
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


class OffsetTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        hint_label = QLabel("Выберите один файл эхограммы и GNSS-трек, затем нажмите "
                             "«Рассчитать смещение».")
        self.calc_btn = QPushButton("Рассчитать смещение")
        self.calc_btn.setEnabled(False)
        self.calc_btn.clicked.connect(self.calc_offset)

        self.content_placeholder = QWidget()

        layout = QVBoxLayout()
        layout.addWidget(hint_label)
        layout.addWidget(self.calc_btn)
        layout.addWidget(self.content_placeholder, 1)
        self.setLayout(layout)

    def calc_offset(self):
        mw = self.main_window
        path = mw.sl2_edit.text()
        if not path or ";" in path:
            QMessageBox.information(
                self, "Расчёт смещения",
                "Расчёт смещения работает только с одним файлом эхограммы (не с "
                "мультиэхограммой) — выберите один файл (кнопка «Обзор…»).")
            return
        self.calc_btn.setEnabled(False)
        mw.statusBar().showMessage("Вычисление смещения…")
        run_async(lambda: compute_time_offset(path, mw.gnss_paths),
                  on_finished=self.on_offset_finished, on_error=self.on_offset_error)

    def on_offset_finished(self, result):
        mw = self.main_window
        mw.statusBar().showMessage(f"Смещение: {result['offset_s']:+.3f} с")
        self.calc_btn.setEnabled(True)
        mw.apply_offset_dates(result)
        self.set_result(mw.basemap_combo.currentText(), result)

    def on_offset_error(self, message):
        mw = self.main_window
        mw.statusBar().showMessage("Ошибка вычисления смещения")
        self.calc_btn.setEnabled(True)
        QMessageBox.critical(self, "Ошибка вычисления смещения", message)

    def set_result(self, basemap, result):
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

        content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.addWidget(report_box)
        content_layout.addLayout(maps_row, 1)
        content.setLayout(content_layout)

        self.layout().replaceWidget(self.content_placeholder, content)
        self.content_placeholder.deleteLater()
        self.content_placeholder = content


class FileTimelineWidget(QWidget):
    """Шкала времени загруженных файлов ровера (зелёный) и базы (красный) —
    видно начало/конец записи каждого файла и их взаимное перекрытие
    (перекрытие нужно для парного расчёта PPK)."""
    ROW_H = 20
    MARGIN = 6
    AXIS_H = 20
    N_TICKS = 5

    def __init__(self):
        super().__init__()
        self.rovers = []
        self.bases = []
        self.setMinimumHeight(self.AXIS_H + 2 * self.ROW_H + 2 * self.MARGIN)

    def set_files(self, rovers, bases):
        self.rovers = rovers
        self.bases = bases
        n_rows = max(1, len(rovers)) + max(1, len(bases))
        self.setMinimumHeight(self.AXIS_H + n_rows * self.ROW_H + 2 * self.MARGIN)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))
        entries = [(e, QColor("#2ecc71")) for e in self.rovers] + \
                  [(e, QColor("#e74c3c")) for e in self.bases]
        painter.setPen(QColor("#cccccc"))
        if not entries:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                              "Файлы не загружены")
            return
        times = [e["start"] for e, _ in entries if e.get("start") is not None]
        times += [e["end"] for e, _ in entries if e.get("end") is not None]
        if not times:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                              "Не удалось определить время файлов")
            return
        t0, t1 = min(times), max(times)
        if t1 <= t0:
            t1 = t0 + 1.0
        x0, x1 = self.MARGIN, self.width() - self.MARGIN
        span = t1 - t0

        def to_x(t):
            return x0 + (t - t0) / span * (x1 - x0)

        # шкала времени сверху: вертикальные засечки на всю высоту + подписи UTC
        painter.setPen(QColor("#555555"))
        for i in range(self.N_TICKS + 1):
            x = int(to_x(t0 + span * i / self.N_TICKS))
            painter.drawLine(x, self.AXIS_H - 4, x, self.height())
        painter.setPen(QColor("#cccccc"))
        for i in range(self.N_TICKS + 1):
            t = t0 + span * i / self.N_TICKS
            x = int(to_x(t))
            label = datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M:%S")
            if i == 0:
                rect_x, align = x, Qt.AlignmentFlag.AlignLeft
            elif i == self.N_TICKS:
                rect_x, align = x - 80, Qt.AlignmentFlag.AlignRight
            else:
                rect_x, align = x - 40, Qt.AlignmentFlag.AlignHCenter
            painter.drawText(rect_x, 0, 80, self.AXIS_H - 4,
                              int(align | Qt.AlignmentFlag.AlignVCenter), label)

        y = self.AXIS_H
        for entry, color in entries:
            name = os.path.basename(entry["path"])
            start, end = entry.get("start"), entry.get("end")
            if entry.get("error"):
                painter.setPen(QColor("#e74c3c"))
                painter.drawText(x0, y + self.ROW_H - 6, f"{name} — ошибка: {entry['error']}")
            elif start is None:
                painter.setPen(QColor("#888888"))
                painter.drawText(x0, y + self.ROW_H - 6, f"{name} — загрузка…")
            else:
                bx0, bx1 = to_x(start), to_x(end)
                painter.fillRect(int(bx0), y + 2, max(2, int(bx1 - bx0)), self.ROW_H - 8, color)
                painter.setPen(QColor("white"))
                painter.drawText(int(bx0) + 3, y + self.ROW_H - 6, name)
            y += self.ROW_H


class PPKTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.rovers = []
        self.bases = []
        self.results = []
        self._pending_pairs = []

        files_group = QGroupBox("Файлы")
        add_rover_btn = QPushButton("Добавить ровер(ы)…")
        add_rover_btn.clicked.connect(self.add_rovers)
        remove_rover_btn = QPushButton("Удалить выбранный")
        remove_rover_btn.clicked.connect(self.remove_selected_rover)
        self.rovers_list = QListWidget()
        self._fix_list_height(self.rovers_list, 2)
        rover_btn_row = QHBoxLayout()
        rover_btn_row.addWidget(add_rover_btn)
        rover_btn_row.addWidget(remove_rover_btn)
        rover_col = QVBoxLayout()
        rover_col.addWidget(label_with_help("Роверы:", PPK_FILES_HELP))
        rover_col.addLayout(rover_btn_row)
        rover_col.addWidget(self.rovers_list)

        add_base_btn = QPushButton("Добавить баз(ы)…")
        add_base_btn.clicked.connect(self.add_bases)
        remove_base_btn = QPushButton("Удалить выбранный")
        remove_base_btn.clicked.connect(self.remove_selected_base)
        self.bases_list = QListWidget()
        self._fix_list_height(self.bases_list, 2)
        base_btn_row = QHBoxLayout()
        base_btn_row.addWidget(add_base_btn)
        base_btn_row.addWidget(remove_base_btn)
        base_col = QVBoxLayout()
        base_col.addWidget(label_with_help("Базы:", PPK_FILES_HELP))
        base_col.addLayout(base_btn_row)
        base_col.addWidget(self.bases_list)

        lists_sep = QFrame()
        lists_sep.setFrameShape(QFrame.Shape.VLine)
        lists_sep.setFrameShadow(QFrame.Shadow.Sunken)

        lists_row = QHBoxLayout()
        lists_row.addLayout(rover_col)
        lists_row.addWidget(lists_sep)
        lists_row.addLayout(base_col)

        self.timeline = FileTimelineWidget()

        timeline_label = QLabel(
            'Таймлайн (<span style="color:#2ecc71;">■</span> — роверы, '
            '<span style="color:#e74c3c;">■</span> — базы):')
        timeline_label.setTextFormat(Qt.TextFormat.RichText)

        files_layout = QVBoxLayout()
        files_layout.addLayout(lists_row)
        files_layout.addWidget(timeline_label)
        files_layout.addWidget(self.timeline)
        files_group.setLayout(files_layout)

        self.systems_edit = QLineEdit("GREC")
        self.systems_edit.setToolTip("Буквы систем ГНСС: G — GPS, R — ГЛОНАСС, "
                                      "E — Galileo, C — BeiDou.")
        self.elmask_spin = QSpinBox()
        self.elmask_spin.setRange(0, 90)
        self.elmask_spin.setValue(10)
        self.elmask_spin.setSuffix(" °")
        self.ar_ratio_spin = QDoubleSpinBox()
        self.ar_ratio_spin.setRange(1.0, 999.0)
        self.ar_ratio_spin.setValue(5.0)
        self.ar_mode_combo = QComboBox()
        self.ar_mode_combo.addItem("Continuous", "continuous")
        self.ar_mode_combo.addItem("Fix and hold", "fix-and-hold")
        self.glo_ar_combo = QComboBox()
        self.glo_ar_combo.addItem("Вкл. (одинаковые приёмники)", "on")
        self.glo_ar_combo.addItem("Autocal", "autocal")
        self.glo_ar_combo.addItem("Выкл.", "off")
        self.frequency_combo = QComboBox()
        self.frequency_combo.addItem("L1", "l1")
        self.frequency_combo.addItem("L1+L2", "l1+l2")
        self.frequency_combo.setCurrentIndex(1)

        rtk_form = QFormLayout()
        rtk_form.addRow(label_with_help("Системы:", PPK_SYSTEMS_HELP), self.systems_edit)
        rtk_form.addRow(label_with_help("Маска возвышения:", PPK_ELMASK_HELP), self.elmask_spin)
        rtk_form.addRow(label_with_help("AR ratio:", PPK_AR_RATIO_HELP), self.ar_ratio_spin)
        rtk_form.addRow(label_with_help("AR mode:", PPK_AR_MODE_HELP), self.ar_mode_combo)
        rtk_form.addRow(label_with_help("ГЛОНАСС AR:", PPK_GLO_AR_HELP), self.glo_ar_combo)
        rtk_form.addRow(label_with_help("Частота:", PPK_FREQUENCY_HELP), self.frequency_combo)

        self.base_pos_single_radio = QRadioButton("Одиночное решение по записи базы")
        self.base_pos_rinexhead_radio = QRadioButton("Из заголовка RINEX")
        self.base_pos_exact_radio = QRadioButton("Точные координаты")
        self.base_pos_single_radio.setChecked(True)
        self.base_pos_group = QButtonGroup(self)
        for btn in (self.base_pos_single_radio, self.base_pos_rinexhead_radio,
                    self.base_pos_exact_radio):
            self.base_pos_group.addButton(btn)
            btn.toggled.connect(self.on_base_pos_toggled)

        self.base_lat_spin = QDoubleSpinBox()
        self.base_lat_spin.setRange(-90.0, 90.0)
        self.base_lat_spin.setDecimals(8)
        self.base_lon_spin = QDoubleSpinBox()
        self.base_lon_spin.setRange(-180.0, 180.0)
        self.base_lon_spin.setDecimals(8)
        self.base_h_spin = QDoubleSpinBox()
        self.base_h_spin.setRange(-1000.0, 10000.0)
        self.base_h_spin.setDecimals(3)
        self.base_h_spin.setSuffix(" м")
        base_coords_form = QFormLayout()
        base_coords_form.addRow("Широта:", self.base_lat_spin)
        base_coords_form.addRow("Долгота:", self.base_lon_spin)
        base_coords_form.addRow("Высота:", self.base_h_spin)
        self.on_base_pos_toggled()

        base_pos_group_box = QGroupBox("Координаты базы")
        base_pos_layout = QVBoxLayout()
        base_pos_layout.addWidget(self.base_pos_single_radio)
        base_pos_layout.addWidget(self.base_pos_rinexhead_radio)
        base_pos_layout.addWidget(self.base_pos_exact_radio)
        base_pos_layout.addLayout(base_coords_form)
        base_pos_group_box.setLayout(base_pos_layout)

        rtk_group = QGroupBox("Настройки RTK")
        rtk_layout = QVBoxLayout()
        rtk_layout.addLayout(rtk_form)
        rtk_layout.addWidget(base_pos_group_box)
        rtk_group.setLayout(rtk_layout)

        self.workdir_edit = QLineEdit()
        workdir_btn = QPushButton("Обзор…")
        workdir_btn.clicked.connect(self.pick_workdir)
        workdir_row = QHBoxLayout()
        workdir_row.addWidget(self.workdir_edit, 1)
        workdir_row.addWidget(workdir_btn)
        workdir_form = QFormLayout()
        workdir_form.addRow("Папка результатов:", workdir_row)

        self.run_btn = QPushButton("Запустить PPK")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run_ppk)
        self.status_label = QLabel("")

        self.results_list = QListWidget()
        self._resize_results_list()
        self.results_list.currentRowChanged.connect(self.show_result_details)
        self.details_view = QTextEdit(readOnly=True)
        self.details_view.setStyleSheet("font-family: Consolas, monospace;")
        self.details_view.setPlaceholderText(
            "Выберите пару в списке слева, чтобы увидеть подробности RINEX "
            "(системы, спутники, координаты приёмника и т.д.).")
        use_track_btn = QPushButton("Использовать выбранный результат как ровер")
        use_track_btn.clicked.connect(self.use_selected_as_track)
        self.map_view = new_map_view()
        self.map_view.setMinimumHeight(220)

        # Низ вкладки — три колонки: слева настройки (25%), посередине пары +
        # карта (50%), справа подробности (25%).
        settings_col = QVBoxLayout()
        settings_col.addWidget(rtk_group)
        settings_col.addLayout(workdir_form)
        settings_col.addWidget(self.run_btn)
        settings_col.addWidget(self.status_label)
        settings_col.addStretch(1)
        self.settings_widget = QWidget()
        self.settings_widget.setLayout(settings_col)

        pairs_group = QGroupBox("Результат (по парам ровер+база)")
        pairs_layout = QVBoxLayout()
        pairs_layout.addWidget(self.results_list)
        pairs_layout.addWidget(QLabel("Карта (трек ровера — зелёный, база — флажок):"))
        pairs_layout.addWidget(self.map_view, 1)
        pairs_layout.addWidget(use_track_btn)
        pairs_group.setLayout(pairs_layout)

        details_group = QGroupBox("Подробности")
        details_layout = QVBoxLayout()
        details_layout.addWidget(self.details_view, 1)
        details_group.setLayout(details_layout)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.settings_widget)
        bottom_row.addWidget(pairs_group, 2)
        bottom_row.addWidget(details_group, 1)

        layout = QVBoxLayout()
        layout.addWidget(files_group)
        layout.addLayout(bottom_row, 1)
        self.setLayout(layout)

        self._refresh_lists()

    def resizeEvent(self, event):
        """Настройки — ровно 25% ширины вкладки: при stretch-факторах в
        bottom_row реальная ширина колонки настроек «плавает» из-за
        собственного размера её содержимого (поля/радиокнопки RTK), поэтому
        фиксируем её явным пересчётом при каждом изменении размера окна."""
        super().resizeEvent(event)
        if self.width() > 0:
            self.settings_widget.setFixedWidth(int(self.width() * 0.25))

    @staticmethod
    def _fix_list_height(list_widget, rows):
        row_h = list_widget.sizeHintForRow(0) if list_widget.count() else 24
        list_widget.setFixedHeight(row_h * rows + 2 * list_widget.frameWidth() + 4)

    def _resize_results_list(self):
        """Список пар — ровно 2 строки, остальное место в этой колонке
        отдаётся карте (см. pairs_layout)."""
        self._fix_list_height(self.results_list, 2)

    def on_base_pos_toggled(self, *_args):
        exact = self.base_pos_exact_radio.isChecked()
        self.base_lat_spin.setEnabled(exact)
        self.base_lon_spin.setEnabled(exact)
        self.base_h_spin.setEnabled(exact)

    def add_rovers(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Добавить ровер(ы)", self.main_window.last_dir(),
            "GNSS (*.ubx *.obs *.rnx *.??o *.gz);;Все файлы (*)")
        for path in paths:
            self.main_window.remember_dir(path)
            self._add_file(self.rovers, path)

    def add_bases(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Добавить баз(ы)", self.main_window.last_dir(),
            "GNSS (*.ubx *.obs *.rnx *.??o *.gz);;Все файлы (*)")
        for path in paths:
            self.main_window.remember_dir(path)
            self._add_file(self.bases, path)

    def remove_selected_rover(self):
        row = self.rovers_list.currentRow()
        if 0 <= row < len(self.rovers):
            del self.rovers[row]
            self._refresh_lists()

    def remove_selected_base(self):
        row = self.bases_list.currentRow()
        if 0 <= row < len(self.bases):
            del self.bases[row]
            self._refresh_lists()

    def add_file_unique(self, entries, path):
        """Как _add_file, но не дублирует уже присутствующий по пути файл —
        для автопереноса ровера/базы, выбранных в шапке окна."""
        if any(e["path"] == path for e in entries):
            return
        self._add_file(entries, path)

    def _add_file(self, entries, path):
        entry = dict(path=path, start=None, end=None, error=None)
        entries.append(entry)
        self._refresh_lists()
        run_async(lambda: self._probe_file(path),
                  on_finished=lambda times: self._on_probed(entry, times, None),
                  on_error=lambda msg: self._on_probed(entry, None, msg))

    @staticmethod
    def _probe_file(path):
        if is_rinex(Path(path)):
            # уже готовый RINEX OBS (.obs/.rnx или годовой .NNo, напр. с приёмников
            # EFT/ComNav и т.п., сконвертированный сторонним ПО) — не .ubx и не
            # распознаётся эвристикой read_gnss(fmt="auto"), поэтому отдельная ветка.
            info = rinex_obs_info(path)
            if info.get("t_first") is None or info.get("t_last") is None:
                raise ValueError("Не удалось определить период записи RINEX "
                                 "(TIME OF FIRST/LAST OBS и эпохи не найдены)")
            return info["t_first"].timestamp(), info["t_last"].timestamp()
        if path.lower().endswith(".ubx"):
            g, _info = read_ubx(path)
        else:
            g, _info = read_gnss(path, fmt="auto")
        return float(g["t"][0]), float(g["t"][-1])

    def _on_probed(self, entry, times, error):
        if error:
            entry["error"] = error
        else:
            entry["start"], entry["end"] = times
        self._refresh_lists()

    def _entry_label(self, e):
        name = os.path.basename(e["path"])
        if e["error"]:
            return f"{name} — ошибка: {e['error']}"
        if e["start"] is None:
            return f"{name} — загрузка…"
        dur_min = (e["end"] - e["start"]) / 60.0
        start_str = datetime.fromtimestamp(e["start"], timezone.utc).strftime("%d.%m.%Y %H:%M:%S")
        return f"{name} — {start_str} UTC, {dur_min:.1f} мин"

    def _refresh_lists(self):
        self.rovers_list.clear()
        self.rovers_list.addItems([self._entry_label(e) for e in self.rovers])
        self._fix_list_height(self.rovers_list, 2)
        self.bases_list.clear()
        self.bases_list.addItems([self._entry_label(e) for e in self.bases])
        self._fix_list_height(self.bases_list, 2)
        self.timeline.set_files(self.rovers, self.bases)
        self.run_btn.setEnabled(bool(self.rovers) and bool(self.bases))
        if self.rovers and not self.workdir_edit.text():
            base, _ext = os.path.splitext(self.rovers[0]["path"])
            self.workdir_edit.setText(base + "_ppk_out")

    def build_config(self):
        base_pos = "single"
        if self.base_pos_rinexhead_radio.isChecked():
            base_pos = "rinexhead"
        elif self.base_pos_exact_radio.isChecked():
            base_pos = (self.base_lat_spin.value(), self.base_lon_spin.value(),
                        self.base_h_spin.value())
        return PPKConfig(systems=self.systems_edit.text().strip() or "GREC",
                         elmask_deg=float(self.elmask_spin.value()),
                         ar_ratio=self.ar_ratio_spin.value(),
                         ar_mode=self.ar_mode_combo.currentData(),
                         glo_ar=self.glo_ar_combo.currentData(),
                         frequency=self.frequency_combo.currentData(),
                         base_pos=base_pos)

    def pick_workdir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Папка результатов PPK", self.workdir_edit.text())
        if path:
            self.workdir_edit.setText(path)

    def find_pairs(self):
        """Для каждого ровера ищет базу с наибольшим перекрытием по времени —
        расчёт PPK возможен только там, где записи ровера и базы велись
        одновременно. Дополнительно: запись ровера должна быть короче записи
        базы — иначе база не покрывает весь период съёмки ровера целиком
        (часть точек останется без поправок), это чаще всего ошибка выбора
        файлов, поэтому пара пропускается с предупреждением, а не считается."""
        pairs, skipped = [], []
        for rover in self.rovers:
            if rover.get("start") is None:
                skipped.append((rover, "не удалось определить период записи"))
                continue
            best, best_overlap = None, 0.0
            for base in self.bases:
                if base.get("start") is None:
                    continue
                overlap = min(rover["end"], base["end"]) - max(rover["start"], base["start"])
                if overlap > best_overlap:
                    best, best_overlap = base, overlap
            if best is None:
                skipped.append((rover, "нет базы, перекрывающейся по времени"))
                continue
            rover_dur = rover["end"] - rover["start"]
            base_dur = best["end"] - best["start"]
            if rover_dur >= base_dur:
                skipped.append((rover, (
                    f"запись ровера ({rover_dur / 60:.1f} мин) не короче записи "
                    f"базы «{os.path.basename(best['path'])}» ({base_dur / 60:.1f} мин) — "
                    "база должна покрывать весь период съёмки ровера")))
                continue
            pairs.append((rover, best))
        return pairs, skipped

    def run_ppk(self):
        if not self.main_window.settings.value("rtklib_dir", "").strip():
            QMessageBox.warning(
                self, "PPK", "Папка RTKLIB не указана — задайте её на вкладке "
                "«Настройки» (группа «PPK»), затем повторите запуск.")
            return
        pairs, skipped = self.find_pairs()
        self.results = [dict(rover=rover, base=None, result=None, error=reason)
                         for rover, reason in skipped]
        if not pairs:
            self._show_results()
            QMessageBox.information(
                self, "PPK", "Нет пар ровер+база, перекрывающихся по времени — "
                "проверьте таймлайн и загруженные файлы.")
            return
        self._pending_pairs = list(pairs)
        self._pending_total = len(pairs)
        self.run_btn.setEnabled(False)
        self.results_list.clear()
        self.results_list.addItem("Обработка…")
        self._process_next_pair()

    def _process_next_pair(self):
        if not self._pending_pairs:
            self.run_btn.setEnabled(True)
            self.status_label.setText(f"Готово: обработано пар {len(self.results)}")
            self._show_results()
            return
        rover, base = self._pending_pairs.pop(0)
        left = len(self._pending_pairs) + 1
        n = self._pending_total - left + 1
        pct = round(100 * (n - 1) / self._pending_total)
        self.status_label.setText(
            f"Обработка… {pct}% (пара {n} из {self._pending_total}): "
            f"{os.path.basename(rover['path'])} + {os.path.basename(base['path'])} "
            f"(может занять несколько минут)…")
        rtklib_dir = self.main_window.settings.value("rtklib_dir", "").strip() or None
        base_workdir = self.workdir_edit.text().strip()
        if not base_workdir:
            base_noext, _ext = os.path.splitext(self.rovers[0]["path"])
            base_workdir = base_noext + "_ppk_out"
            self.workdir_edit.setText(base_workdir)
        rover_stem = os.path.splitext(os.path.basename(rover["path"]))[0]
        base_stem = os.path.splitext(os.path.basename(base["path"]))[0]
        workdir = os.path.join(base_workdir, f"{rover_stem}__{base_stem}")
        config = self.build_config()
        run_async(lambda: self._run_pair(rover["path"], base["path"], workdir, rtklib_dir, config),
                  on_finished=lambda payload: self._on_pair_done(rover, base, payload, None),
                  on_error=lambda msg: self._on_pair_done(rover, base, None, msg))

    @staticmethod
    def _run_pair(rover_path, base_path, workdir, rtklib_dir, config):
        result = PPKProcessor(rtklib_dir).run(rover_path, base_path, workdir, config)
        rover_stem = os.path.splitext(os.path.basename(rover_path))[0]
        base_stem = os.path.splitext(os.path.basename(base_path))[0]
        rover_candidate = os.path.join(workdir, f"{rover_stem}_rover.obs")
        base_candidate = os.path.join(workdir, f"{base_stem}_base.obs")
        rover_info = PPKTab._safe_rinex_info(rover_candidate, rover_path)
        base_info = PPKTab._safe_rinex_info(base_candidate, base_path)
        return result, rover_info, base_info

    @staticmethod
    def _safe_rinex_info(candidate, fallback):
        # convbin пишет <имя>_rover.obs/<имя>_base.obs в workdir пары, если
        # исходник ещё не был RINEX (обычный случай для .ubx, см. _to_rinex в
        # ppk_module.py); если исходник УЖЕ был RINEX (в т.ч. .gz) —
        # PPKProcessor его не копирует под этим именем, тогда сведения читаем
        # прямо из исходника (rinex_obs_info сам умеет читать .gz).
        path = candidate if os.path.exists(candidate) else fallback
        try:
            return rinex_obs_info(path)
        except Exception as e:
            return dict(path=path, error=f"{type(e).__name__}: {e}")

    def _on_pair_done(self, rover, base, payload, error):
        if error:
            self.results.append(dict(rover=rover, base=base, result=None,
                                     rover_info=None, base_info=None, error=error))
        else:
            result, rover_info, base_info = payload
            self.results.append(dict(rover=rover, base=base, result=result,
                                     rover_info=rover_info, base_info=base_info, error=None))
        self._process_next_pair()

    @staticmethod
    def _event_count(item):
        """Суммарное число временных меток (эпох-событий RINEX, flag=5) по
        ровepy+базе пары — обычно это внешние отметки времени, например от
        синхроимпульса эхолота, если приёмник их пишет."""
        n = 0
        for key in ("rover_info", "base_info"):
            info = item.get(key)
            if info and not info.get("error"):
                n += info.get("n_events", 0)
        return n

    def _show_results(self):
        self.results_list.clear()
        for item in self.results:
            rover_name = os.path.basename(item["rover"]["path"])
            base_name = os.path.basename(item["base"]["path"]) if item["base"] else "—"
            if item["error"]:
                text = f"{rover_name} + {base_name}: ОШИБКА — {item['error']}"
            else:
                st = item["result"].stats
                text = (f"{rover_name} + {base_name}: fix {st['fix_%']:.1f} %, "
                        f"float {st['float_%']:.1f} %, single {st['single_%']:.1f} %")
            n_events = self._event_count(item)
            if n_events > 0:
                text += f"  ⚠ временных меток: {n_events}"
                list_item = QListWidgetItem(warning_icon(), text)
            else:
                list_item = QListWidgetItem(text)
            self.results_list.addItem(list_item)
        self._resize_results_list()

    def show_result_details(self, row):
        if not (0 <= row < len(self.results)):
            self.details_view.setPlainText("")
            load_html(self.map_view, build_ppk_map_html(
                self.main_window.basemap_combo.currentText(), "", [], [], "", None, None),
                "ppk_result")
            return
        item = self.results[row]
        self.details_view.setHtml(self.format_pair_details(item))

        rover_name = os.path.basename(item["rover"]["path"])
        base_name = os.path.basename(item["base"]["path"]) if item["base"] else ""
        rover_lat, rover_lon = [], []
        if item["result"] is not None:
            track = item["result"].track
            rover_lat, rover_lon = track["lat"].tolist(), track["lon"].tolist()
        base_lat = base_lon = None
        base_info = item.get("base_info")
        if base_info and base_info.get("wgs84"):
            base_lat, base_lon, _h = base_info["wgs84"]
        html = build_ppk_map_html(self.main_window.basemap_combo.currentText(),
                                  rover_name, rover_lat, rover_lon, base_name, base_lat, base_lon)
        load_html(self.map_view, html, "ppk_result")

    def format_pair_details(self, item):
        esc = html.escape
        if item["error"]:
            return f"Ошибка расчёта пары: {esc(item['error'])}"
        lines = []
        if item["result"] is not None:
            st = item["result"].stats
            sdu = st.get("sdu_fix_median_m")
            ppk_line = (f"fix {st['fix_%']:.1f}%, float {st['float_%']:.1f}%, "
                       f"single {st['single_%']:.1f}%"
                       + (f", СКО фикса (верт.) {sdu:.3f} м" if sdu is not None else ""))
            lines.append(f"<b>PPK:</b> {esc(ppk_line)}")
        n_events = self._event_count(item)
        if n_events > 0:
            lines.append(f"⚠ Временных меток (event, flag=5): {n_events} — суммарно по роверу "
                        f"и базе, см. разбивку по файлам ниже.")
        lines.append("")
        lines.append("<b>=== Ровер ===</b>")
        lines.append(self._format_rinex_info(item.get("rover_info")))
        lines.append("")
        lines.append("<b>=== База ===</b>")
        lines.append(self._format_rinex_info(item.get("base_info")))
        return "<br>".join(lines)

    @staticmethod
    def _format_rinex_info(info):
        esc = html.escape
        if not info:
            return "(нет данных)"
        if info.get("error"):
            return f"Не удалось прочитать RINEX: {esc(info['error'])}"
        lines = [f"{esc(os.path.basename(info['path']))}  ({info['size_bytes'] / 1e6:.1f} МБ)"]
        t1, t2 = info.get("t_first"), info.get("t_last")
        if t1 and t2:
            lines.append(f"<b>Период:</b> {t1:%Y-%m-%d %H:%M:%S} — {t2:%Y-%m-%d %H:%M:%S} UTC")
            total_s = int((t2 - t1).total_seconds())
            h, rem = divmod(total_s, 3600)
            m, s = divmod(rem, 60)
            interval = info.get("interval")
            interval_str = f"{interval:.2f} с" if interval is not None else "—"
            lines.append(f"<b>Длительность:</b> {h:02d}:{m:02d}:{s:02d}   "
                        f"Интервал: {interval_str}   Эпох: {info.get('n_epochs', 0)}   "
                        f"Спецотметок (event): {info.get('n_events', 0)}")
        for sysinfo in info.get("systems", {}).values():
            bands = " ".join(sysinfo["bands"])
            lines.append(f"{esc(sysinfo['name'])}[{sysinfo['n_sat']:02d}]: {esc(bands)}")
        xyz = info.get("approx_xyz")
        if xyz:
            lines.append(f"<b>ECEF:</b> {xyz[0]:.3f} {xyz[1]:.3f} {xyz[2]:.3f}")
        wgs = info.get("wgs84")
        if wgs:
            lat, lon, h = wgs
            lines.append(f"<b>WGS84:</b> {lat:.8f} {lon:.8f}   выс. {h:.3f} м")
        version = info.get("version")
        lines.append(f"<b>Приёмник:</b> {esc(info.get('rec_type') or '—')}   "
                    f"RINEX: {version if version is not None else '—'}")
        return "<br>".join(lines)

    def use_selected_as_track(self):
        row = self.results_list.currentRow()
        if not (0 <= row < len(self.results)):
            return
        item = self.results[row]
        if item["error"] or item["result"] is None:
            QMessageBox.information(self, "PPK", "У выбранной пары нет результата (ошибка расчёта).")
            return
        self.main_window.set_gnss_track([str(item["result"].pos_file)])


class ExportTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        sync_group = QGroupBox("Синхронизация")
        self.manual_offset_check = QCheckBox("Задать смещение вручную (иначе — автопоиск)")
        self.manual_offset_check.toggled.connect(self.on_manual_offset_toggled)
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setSuffix(" с")
        self.drift_spin = QDoubleSpinBox()
        self.drift_spin.setRange(-1000.0, 1000.0)
        self.drift_spin.setSuffix(" ppm")
        self.latency_spin = QDoubleSpinBox()
        self.latency_spin.setRange(-100.0, 100.0)
        self.latency_spin.setDecimals(2)
        self.latency_spin.setSuffix(" с")
        self.auto_latency_check = QCheckBox("Подобрать задержку глубины автоматически")
        self.auto_latency_check.toggled.connect(self.on_auto_latency_toggled)
        sync_form = QFormLayout()
        sync_form.addRow(self.manual_offset_check)
        sync_form.addRow("Смещение:", self.offset_spin)
        sync_form.addRow("Дрейф часов:", self.drift_spin)
        sync_form.addRow("Задержка глубины:", self.latency_spin)
        sync_form.addRow(self.auto_latency_check)
        sync_group.setLayout(sync_form)
        self.on_manual_offset_toggled(False)
        self.on_auto_latency_toggled(False)

        geo_group = QGroupBox("Геометрия")
        self.lever_fwd_spin = QDoubleSpinBox()
        self.lever_fwd_spin.setRange(-100.0, 100.0)
        self.lever_fwd_spin.setSuffix(" м")
        self.lever_right_spin = QDoubleSpinBox()
        self.lever_right_spin.setRange(-100.0, 100.0)
        self.lever_right_spin.setSuffix(" м")
        lever_row = QHBoxLayout()
        lever_row.addWidget(QLabel("Вперёд:"))
        lever_row.addWidget(self.lever_fwd_spin)
        lever_row.addWidget(QLabel("Вправо:"))
        lever_row.addWidget(self.lever_right_spin)
        self.draft_spin = QDoubleSpinBox()
        self.draft_spin.setRange(-100.0, 100.0)
        self.draft_spin.setSuffix(" м")
        self.antenna_height_check = QCheckBox("Считать отметку дна (указана высота антенны)")
        self.antenna_height_check.toggled.connect(self.on_antenna_height_toggled)
        self.antenna_height_spin = QDoubleSpinBox()
        self.antenna_height_spin.setRange(-100.0, 100.0)
        self.antenna_height_spin.setDecimals(3)
        self.antenna_height_spin.setSuffix(" м")
        geo_form = QFormLayout()
        geo_form.addRow("Датчик относительно антенны:", lever_row)
        geo_form.addRow("Осадка датчика:", self.draft_spin)
        geo_form.addRow(self.antenna_height_check)
        geo_form.addRow("Высота антенны:", self.antenna_height_spin)
        geo_group.setLayout(geo_form)
        self.on_antenna_height_toggled(False)

        filt_group = QGroupBox("Фильтрация и грид")
        self.min_fix_combo = QComboBox()
        self.min_fix_combo.addItem("Любое", "any")
        self.min_fix_combo.addItem("Float и лучше", "float")
        self.min_fix_combo.addItem("Только Fix", "fix")
        self.min_depth_spin = QDoubleSpinBox()
        self.min_depth_spin.setRange(0.0, 1000.0)
        self.min_depth_spin.setValue(0.2)
        self.min_depth_spin.setSuffix(" м")
        self.max_depth_spin = QDoubleSpinBox()
        self.max_depth_spin.setRange(0.0, 1000.0)
        self.max_depth_spin.setValue(150.0)
        self.max_depth_spin.setSuffix(" м")
        depth_range_row = QHBoxLayout()
        depth_range_row.addWidget(QLabel("Мин:"))
        depth_range_row.addWidget(self.min_depth_spin)
        depth_range_row.addWidget(QLabel("Макс:"))
        depth_range_row.addWidget(self.max_depth_spin)
        self.grid_cell_spin = QDoubleSpinBox()
        self.grid_cell_spin.setRange(0.01, 1000.0)
        self.grid_cell_spin.setValue(0.5)
        self.grid_cell_spin.setSuffix(" м")
        self.no_grid_check = QCheckBox("Не строить грид")
        filt_form = QFormLayout()
        filt_form.addRow("Мин. качество GNSS:", self.min_fix_combo)
        filt_form.addRow("Диапазон глубин:", depth_range_row)
        filt_form.addRow("Ячейка грида:", self.grid_cell_spin)
        filt_form.addRow(self.no_grid_check)
        filt_group.setLayout(filt_form)

        crs_group = QGroupBox("Система координат (для колонок E/N и грида)")
        self.crs_combo = QComboBox()
        for label, value in COORD_SYSTEMS.items():
            self.crs_combo.addItem(label, value)
        self._load_custom_crs()
        add_crs_btn = QPushButton("Добавить по EPSG-коду…")
        add_crs_btn.clicked.connect(self.add_custom_crs)
        crs_row = QHBoxLayout()
        crs_row.addWidget(self.crs_combo, 1)
        crs_row.addWidget(add_crs_btn)
        crs_layout = QVBoxLayout()
        crs_layout.addLayout(crs_row)
        crs_group.setLayout(crs_layout)

        self.workdir_edit = QLineEdit()
        workdir_btn = QPushButton("Обзор…")
        workdir_btn.clicked.connect(self.pick_workdir)
        workdir_row = QHBoxLayout()
        workdir_row.addWidget(self.workdir_edit, 1)
        workdir_row.addWidget(workdir_btn)
        workdir_form = QFormLayout()
        workdir_form.addRow("Папка результатов:", workdir_row)

        self.run_btn = QPushButton("Запустить расчёт")
        self.run_btn.clicked.connect(self.run_export)
        self.status_label = QLabel("")

        self.result_group = QGroupBox("Результат")
        self.summary_points_label = QLabel("—")
        self.summary_depth_label = QLabel("—")
        self.summary_offset_label = QLabel("—")
        self.summary_latency_label = QLabel("—")
        self.summary_crs_label = QLabel("—")
        summary_form = QFormLayout()
        summary_form.addRow("Точек:", self.summary_points_label)
        summary_form.addRow("Диапазон глубин:", self.summary_depth_label)
        summary_form.addRow("Смещение:", self.summary_offset_label)
        summary_form.addRow("Задержка глубины:", self.summary_latency_label)
        summary_form.addRow("Система координат:", self.summary_crs_label)
        self.files_list = QListWidget()
        self.files_list.itemDoubleClicked.connect(self.open_file_location)
        result_layout = QVBoxLayout()
        result_layout.addLayout(summary_form)
        result_layout.addWidget(QLabel("Файлы (двойной клик — открыть папку):"))
        result_layout.addWidget(self.files_list)
        self.result_group.setLayout(result_layout)
        self.result_group.setVisible(False)

        layout = QVBoxLayout()
        layout.addWidget(sync_group)
        layout.addWidget(geo_group)
        layout.addWidget(filt_group)
        layout.addWidget(crs_group)
        layout.addLayout(workdir_form)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.status_label)
        layout.addWidget(self.result_group, 1)
        self.setLayout(layout)
        self.load_geometry_defaults()

    def load_geometry_defaults(self):
        """Подставляет вынос/осадку датчика, сохранённые через Настройки →
        «Смещение антенн…», как умолчания в группу «Геометрия» — их всё равно
        можно переопределить перед конкретным расчётом."""
        settings = self.main_window.settings
        self.lever_fwd_spin.setValue(float(settings.value("geom_lever_fwd", 0.0)))
        self.lever_right_spin.setValue(float(settings.value("geom_lever_right", 0.0)))
        self.draft_spin.setValue(float(settings.value("geom_draft", 0.0)))
        self.antenna_height_spin.setValue(float(settings.value("geom_antenna_height", 0.0)))

    def on_manual_offset_toggled(self, checked):
        self.offset_spin.setEnabled(checked)
        self.drift_spin.setEnabled(checked)

    def on_auto_latency_toggled(self, checked):
        self.latency_spin.setEnabled(not checked)

    def on_antenna_height_toggled(self, checked):
        self.antenna_height_spin.setEnabled(checked)

    def _load_custom_crs(self):
        raw = self.main_window.settings.value("custom_crs_list", [])
        for entry in raw if isinstance(raw, list) else [raw]:
            label, _sep, code = str(entry).partition("|")
            if label and code:
                self.crs_combo.addItem(label, code)

    def add_custom_crs(self):
        code, ok = QInputDialog.getText(
            self, "Добавить систему координат по EPSG",
            "EPSG-код (например, 32642):")
        if not ok or not code.strip():
            return
        code = code.strip()
        try:
            crs = CRS.from_epsg(int(code))
        except Exception:
            QMessageBox.warning(self, "Система координат",
                                 f"Не удалось найти систему координат EPSG:{code}.")
            return
        label = f"{crs.name} (EPSG:{code})"
        self.crs_combo.addItem(label, code)
        self.crs_combo.setCurrentIndex(self.crs_combo.count() - 1)
        raw = self.main_window.settings.value("custom_crs_list", [])
        entries = list(raw) if isinstance(raw, list) else ([raw] if raw else [])
        entries.append(f"{label}|{code}")
        self.main_window.settings.setValue("custom_crs_list", entries)

    def pick_workdir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Папка результатов экспорта", self.workdir_edit.text())
        if path:
            self.workdir_edit.setText(path)

    def run_export(self):
        mw = self.main_window
        sl2 = mw.sl2_edit.text()
        if not sl2 or ";" in sl2:
            QMessageBox.information(
                self, "Экспорт",
                "Экспорт работает только с одним файлом эхограммы (не с "
                "мультиэхограммой) — выберите один файл (кнопка «Обзор…»).")
            return
        if len(mw.gnss_paths) != 1:
            QMessageBox.information(
                self, "Экспорт",
                "Экспорт поддерживает ровно один GNSS-трек — загрузите один файл "
                "(не мультитрек).")
            return
        workdir = self.workdir_edit.text().strip()
        if not workdir:
            base, _ext = os.path.splitext(sl2)
            workdir = base + "_out"
            self.workdir_edit.setText(workdir)

        ns = build_parser().parse_args([sl2, mw.gnss_paths[0]])
        ns.out = workdir
        ns.offset = self.offset_spin.value() if self.manual_offset_check.isChecked() else None
        ns.drift_ppm = self.drift_spin.value() if self.manual_offset_check.isChecked() else 0.0
        ns.latency = self.latency_spin.value()
        ns.auto_latency = self.auto_latency_check.isChecked()
        ns.lever = [self.lever_fwd_spin.value(), self.lever_right_spin.value()]
        ns.draft = self.draft_spin.value()
        ns.antenna_height = (self.antenna_height_spin.value()
                              if self.antenna_height_check.isChecked() else None)
        ns.min_fix = self.min_fix_combo.currentData()
        ns.min_depth = self.min_depth_spin.value()
        ns.max_depth = self.max_depth_spin.value()
        ns.grid_cell = self.grid_cell_spin.value()
        ns.no_grid = self.no_grid_check.isChecked()
        ns.crs = self.crs_combo.currentData()
        ns.crs_label = self.crs_combo.currentText()

        self.run_btn.setEnabled(False)
        self.status_label.setText("Выполняется…")
        self.result_group.setVisible(False)
        run_async(lambda: sl2sync_run(ns),
                  on_finished=self.on_export_finished, on_error=self.on_export_error)

    def on_export_finished(self, result):
        self.run_btn.setEnabled(True)
        self.status_label.setText("Готово")
        res = result["res"]
        self.summary_points_label.setText(str(res.get("points", "—")))
        dr = res.get("depth_range_m")
        self.summary_depth_label.setText(f"{dr[0]:.2f}–{dr[1]:.2f} м" if dr else "—")
        self.summary_offset_label.setText(f"{res.get('offset_s', 0.0):+.3f} с")
        self.summary_latency_label.setText(f"{res.get('latency_s', 0.0):+.2f} с")
        self.summary_crs_label.setText(str(res.get("crs", "—")))
        self.files_list.clear()
        self.files_list.addItems(result["files"])
        self.result_group.setVisible(True)

    def on_export_error(self, message):
        self.run_btn.setEnabled(True)
        self.status_label.setText("Ошибка")
        QMessageBox.critical(self, "Ошибка экспорта", message)

    def open_file_location(self, item):
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(os.path.abspath(item.text()))))


HELP_HTML = """
<h1>ОМДЖЕТ Гидро — инструкция</h1>
<p>Промер глубин водоёма эхолотом Lowrance (.sl2) с точной привязкой по GNSS.
Обычный порядок работы: <b>Карта</b> → при необходимости <b>PPK</b> →
<b>Расчёт смещения</b> → <b>Построение изобат</b> / <b>Донные отложения</b> →
<b>Экспорт данных</b>.</p>

<h2>Шапка окна</h2>
<p>Общие для всех вкладок поля вверху:</p>
<ul>
<li><b>Эхограмма</b> — запись эхолота Lowrance (.sl2). Кнопка
«Мультиэхограмма» загружает несколько файлов одного выхода подряд и
объединяет их на одной карте.</li>
<li><b>Ровер</b> — точный GNSS-трек (NMEA-лог, RTKLIB .pos, CSV или UBX) для
расчёта смещения; это же поле служит ровером для вкладки «PPK». Файл .ubx,
выбранный здесь, автоматически добавляется и в список роверов PPK.</li>
<li><b>База</b> — файл базовой (стационарной) станции, нужен только для
PPK; автоматически добавляется в список баз PPK.</li>
</ul>
<p>Диалоги выбора файлов запоминают последнюю открытую папку. Рядом с полями
выбора файлов — значки «?» с описанием принимаемых форматов.</p>

<h2>Карта</h2>
<p>Трек эхолота на подложке (Google Спутник/Карта) с сводкой (даты, скорость,
протяжённость, глубины), обрезкой начала/конца записи и смещением глубины
(например, компенсация осадки датчика). Раскраска точек — по глубине, по
твёрдости дна (см. «Донные отложения») или без раскраски. Флажки отмечают
начало/конец трека и точку максимальной глубины; включаемое свечение вдоль
трека показывает скорость.</p>

<h2>Эхограмма</h2>
<p>Водопад сонара выбранного файла эхограммы (столбец — пинг, строка —
глубина), с линейкой глубин слева и осью времени/расстояния снизу. Таймлайн
профиля глубины под эхограммой размечен так же, как таймлайн на «Карте»
(линии сетки и подписи времени/расстояния); колесо мыши над ним меняет
масштаб эхограммы — как и колесо над самим изображением.</p>
<ul>
<li><b>Режим заметок</b> — включите и кликните по эхограмме, чтобы поставить
заметку с текстом в этом месте; список заметок и текст — справа, можно
редактировать и удалять. Под ними — мини-карта с треком: при наведении на
эхограмму на ней показывается жёлтой точкой соответствующее место на воде.</li>
<li><b>Линейка</b> — включите и кликните по двум точкам: покажет расстояние
между ними и по глубине, и по времени/дистанции одновременно.</li>
<li><b>Сохранить заметки / Загрузить заметки</b> — в отдельный JSON-файл рядом
с эхограммой.</li>
</ul>

<h2>PPK</h2>
<p>Постобработка (Post-Processing Kinematic) GNSS-трека по паре ровер+база
через RTKLIB demo5 — вместо одиночного/DGPS-фикса получается решение с
точностью до сантиметров (RTK FIX).</p>
<ul>
<li>Загрузите один или несколько файлов роверов и баз кнопками «Добавить
ровер(ы)…» / «Добавить баз(ы)…» — принимаются .ubx (сырой поток, конвертируется
автоматически) и уже готовый RINEX (.obs/.rnx/.NNo, в т.ч. сжатый в .gz), с
парным NAV-файлом рядом.</li>
<li>Таймлайн под списками показывает период записи каждого файла — зелёным
роверы, красным базы — и их взаимное перекрытие (оно обязательно для
расчёта).</li>
<li>«Запустить PPK» сам находит для каждого ровера базу с максимальным
перекрытием по времени и считает все пары по очереди (запись ровера должна
быть короче записи базы — иначе пара пропускается с предупреждением).</li>
<li>Результат — список пар со статистикой (fix/float/single %), мини-карта
(трек ровера и флажок базы) и подробности по каждому RINEX-файлу пары
(период, системы и спутники, координаты приёмника и т.д.).</li>
<li>«Использовать выбранный результат как ровер» подставляет .pos
выбранной пары в поле «Ровер» шапки окна.</li>
</ul>
<p>Путь к папке RTKLIB (convbin/rnx2rtkp) задаётся один раз в «Настройках».</p>

<h2>Расчёт смещения</h2>
<p>Кнопка «Рассчитать смещение» (нужны один файл эхограммы и один GNSS-трек)
находит временной сдвиг между часами эхолота и GNSS — ту же оценку, что
использует полный расчёт при экспорте (грубое смещение по кросс-корреляции
скорости + уточнение по траектории). Показывает подробный отчёт (смещение,
дрейф часов, СКО позиций) и две карты — трек эхолота и GNSS-трек до и после
синхронизации.</p>

<h2>Построение изобат</h2>
<p>Строит линии постоянной глубины и заливку по точкам трека
(интерполяция + контуры). Карта на этой вкладке обновляется при каждом
переходе на неё; переключатель «Карта» / «3D дно» над ней показывает вместо
2D-контуров тот же грид как трёхмерную поверхность (ЛКМ — вращение, колесо —
зум, ПКМ — сдвиг; ползунок «Преувеличение рельефа» меняет вертикальный
масштаб мгновенно).</p>
<ul>
<li><b>Точки</b> — ручная чистка выбросов промера: «Удалить точки» включает
режим, клик по точке трека на карте исключает её из расчёта (клик ещё раз —
вернуть); «Восстановить все» возвращает все точки разом.</li>
<li><b>Урез</b> — линии уреза воды (граница вода/суша, глубина всегда 0 м) и
линии русла (продолжение интерполяции от ближайших промеров, без выдумывания
глубины). Линий каждого типа может быть несколько. «Загрузить урез» — CSV/
текст (всегда урез) или KML (цвет линии определяет тип: оранжевая/красная —
урез, голубая/синяя — русло). «Нарисовать урез» / «Нарисовать русло» — клик
по карте добавляет точку (концы линий притягиваются друг к другу), ПКМ
завершает линию или удаляет линию под курсором, Ctrl+клик вставляет точку в
линию, Shift+клик удаляет ближайшую точку. Чекбокс «Сглаживание» сглаживает
острые углы нарисованных линий (видно сразу на карте); «Скрыть» прячет линии,
не удаляя точки. Изобаты доходят до нарисованных линий, даже если урез —
открытая линия или короткий фрагмент, включая заливы, куда лодка не
подходила вплотную к берегу.</li>
<li>Настраиваются шаг изобат, ячейка сетки, сглаживание грида, цвет/толщина
линий, подписи (размер и частота) и заливка (прозрачность, число ступеней
перехода цвета — 0 значит один цвет без переходов).</li>
</ul>

<h2>Донные отложения</h2>
<p>Экспериментальная аналитика по индексу твёрдости дна (резкость эхосигнала
+ наличие второго эха) — индекс не откалиброван и не привязан к реальным
типам грунта, это относительный показатель для сравнения точек в пределах
одной записи. Вкладка показывает карту, окрашенную по индексу, гистограмму
распределения, статистику (среднее, медиана, СКО, мин/макс) и корреляцию с
глубиной.</p>

<h2>Экспорт данных</h2>
<p>Полный расчёт — то же самое, что консольная утилита sl2sync.py: синхронизация
времени, геометрия (вынос датчика относительно антенны, осадка, высота
антенны), фильтрация и построение грида. Результат — CSV точек, растровый
грид глубин/отметок дна (.asc и .prj, для ГИС), PNG-отчёт и summary.json.
Систему координат для колонок E/N и грида можно выбрать — местные (МСК)
читаются из файла coord_systems.txt, либо добавить любую по EPSG-коду.</p>

<h2>Настройки</h2>
<p>Окраска глубин и донных отложений (авто/ручной диапазон, число ступеней),
вид трека, временная шкала (время/расстояние), кэш карт, папка RTKLIB (для
PPK), список местных систем координат (для экспорта), геометрия судна
(«Смещение антенн…» — вынос и осадка датчика со схемой-пояснением, значения
становятся умолчанием на вкладке «Экспорт данных»), проверка обновлений.</p>
"""


class HelpTab(QWidget):
    """Встроенная инструкция — текст зашит в код (не читается из README.md),
    чтобы работать одинаково и при запуске из исходников, и в собранном .exe,
    где файла README.md рядом может не быть."""

    def __init__(self):
        super().__init__()
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet("QTextBrowser { background-color: #1e1e1e; color: #e0e0e0; }")
        browser.setHtml(HELP_HTML)
        layout = QVBoxLayout()
        layout.addWidget(browser)
        self.setLayout(layout)


class SettingsTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        settings = main_window.settings
        self.settings = settings

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

        hardness_group = QGroupBox("Окраска донных отложений")
        self.hardness_auto_radio = QRadioButton("Автоматическая (0..1, вся шкала индекса)")
        self.hardness_manual_radio = QRadioButton("Ручная")
        hardness_manual = settings.value("hardness_mode", "auto") == "manual"
        self.hardness_manual_radio.setChecked(hardness_manual)
        self.hardness_auto_radio.setChecked(not hardness_manual)

        self.hardness_min_spin = QDoubleSpinBox()
        self.hardness_min_spin.setRange(0.0, 1.0)
        self.hardness_min_spin.setDecimals(2)
        self.hardness_min_spin.setSingleStep(0.05)
        self.hardness_min_spin.setValue(float(settings.value("hardness_min", 0.0)))
        self.hardness_max_spin = QDoubleSpinBox()
        self.hardness_max_spin.setRange(0.0, 1.0)
        self.hardness_max_spin.setDecimals(2)
        self.hardness_max_spin.setSingleStep(0.05)
        self.hardness_max_spin.setValue(float(settings.value("hardness_max", 1.0)))
        hardness_manual_row = QHBoxLayout()
        hardness_manual_row.addWidget(QLabel("Мин:"))
        hardness_manual_row.addWidget(self.hardness_min_spin)
        hardness_manual_row.addWidget(QLabel("Макс:"))
        hardness_manual_row.addWidget(self.hardness_max_spin)

        self.hardness_steps_spin = QSpinBox()
        self.hardness_steps_spin.setRange(2, 128)
        self.hardness_steps_spin.setValue(int(settings.value("hardness_steps", 32)))
        hardness_steps_row = QHBoxLayout()
        hardness_steps_row.addWidget(QLabel("Число ступеней окраски:"))
        hardness_steps_row.addWidget(self.hardness_steps_spin)

        def update_hardness_manual_enabled():
            self.hardness_min_spin.setEnabled(self.hardness_manual_radio.isChecked())
            self.hardness_max_spin.setEnabled(self.hardness_manual_radio.isChecked())

        self.hardness_auto_radio.toggled.connect(update_hardness_manual_enabled)
        update_hardness_manual_enabled()

        hardness_layout = QVBoxLayout()
        hardness_layout.addWidget(self.hardness_auto_radio)
        hardness_layout.addWidget(self.hardness_manual_radio)
        hardness_layout.addLayout(hardness_manual_row)
        hardness_layout.addLayout(hardness_steps_row)
        hardness_group.setLayout(hardness_layout)

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

        ppk_group = QGroupBox("PPK")
        self.rtklib_dir_edit = QLineEdit(settings.value("rtklib_dir", ""))
        rtklib_btn = QPushButton("Обзор…")
        rtklib_btn.clicked.connect(self.pick_rtklib_dir)
        rtklib_row = QHBoxLayout()
        rtklib_row.addWidget(self.rtklib_dir_edit, 1)
        rtklib_row.addWidget(rtklib_btn)
        ppk_form = QFormLayout()
        ppk_form.addRow(label_with_help(
            "Папка RTKLIB (convbin/rnx2rtkp):",
            "Папка с бинарниками RTKLIB demo5 (convbin.exe, rnx2rtkp.exe), нужна "
            "для расчёта PPK на одноимённой вкладке. Сохраняется сразу же, отдельно "
            "от остальных настроек этой вкладки (кнопка «Применить» тут ни при чём)."),
            rtklib_row)
        ppk_group.setLayout(ppk_form)

        crs_group = QGroupBox("Система координат")
        open_crs_btn = QPushButton("Открыть список…")
        open_crs_btn.clicked.connect(self.open_coord_systems_file)
        crs_layout = QVBoxLayout()
        crs_layout.addWidget(label_with_help(
            "Местные системы координат (для вкладки «Экспорт данных»):",
            "Список читается из текстового файла coord_systems.txt рядом с "
            "программой — кнопка открывает его в обычном текстовом редакторе. "
            "Формат строки как в mapinfow.prj (см. комментарий в начале файла): "
            "\"имя\", 8, 1001, 7, центральный_меридиан, 0, 1, ложное_смещение_E, "
            "ложное_смещение_N. Строку без кавычек в начале файл считает "
            "заголовком раздела — она добавляется к названиям систем ниже неё. "
            "После правки файла перезапустите программу, чтобы изменения "
            "подхватились в списке систем координат."))
        crs_layout.addWidget(open_crs_btn)
        crs_group.setLayout(crs_layout)

        geom_group = QGroupBox("Геометрия судна")
        geom_offset_btn = QPushButton("Смещение антенн…")
        geom_offset_btn.clicked.connect(self.open_antenna_offset)
        geom_layout = QVBoxLayout()
        geom_layout.addWidget(label_with_help(
            "Вынос и осадка датчика эхолота относительно GNSS-антенны:",
            "Открывает окно со схемой и полями выноса/осадки датчика — те же "
            "параметры, что в группе «Геометрия» на вкладке «Экспорт данных» "
            "(--lever/--draft/--antenna-height), но сохраняются как умолчания, "
            "чтобы не вводить заново на каждый расчёт."))
        geom_layout.addWidget(geom_offset_btn)
        geom_group.setLayout(geom_layout)

        update_group = QGroupBox("Обновления")
        version_label = QLabel(f"Текущая версия: {APP_VERSION} (от {APP_VERSION_DATE})")
        update_btn = QPushButton("Проверить обновления")
        update_btn.clicked.connect(lambda: main_window.check_for_updates(silent=False))
        github_btn = QPushButton("Перейти на GitHub")
        github_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(f"https://github.com/{GITHUB_REPO}")))
        update_btn_row = QHBoxLayout()
        update_btn_row.addWidget(update_btn)
        update_btn_row.addWidget(github_btn)
        update_btn_row.addStretch(1)
        update_layout = QVBoxLayout()
        update_layout.addWidget(version_label)
        update_layout.addLayout(update_btn_row)
        update_group.setLayout(update_layout)

        apply_btn = QPushButton("Применить")
        apply_btn.clicked.connect(self.apply_settings)

        def paired_row(left, right):
            row = QHBoxLayout()
            row.addWidget(left, 1)
            row.addWidget(right, 1)
            return row

        layout = QVBoxLayout()
        layout.addLayout(paired_row(depth_group, hardness_group))
        layout.addLayout(paired_row(track_group, axis_group))
        layout.addLayout(paired_row(cache_group, ppk_group))
        layout.addLayout(paired_row(crs_group, geom_group))
        layout.addWidget(update_group)
        layout.addWidget(apply_btn)
        layout.addStretch(1)
        self.setLayout(layout)

    def pick_rtklib_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Папка RTKLIB (convbin/rnx2rtkp)", self.rtklib_dir_edit.text())
        if path:
            self.rtklib_dir_edit.setText(path)
            self.settings.setValue("rtklib_dir", path)

    def open_coord_systems_file(self):
        if not os.path.exists(COORD_SYSTEMS_FILE):
            try:
                with open(COORD_SYSTEMS_FILE, "w", encoding="utf-8") as fh:
                    fh.write(
                        '# Местные системы координат для вкладки «Экспорт данных».\n'
                        '# Формат строки — как в mapinfow.prj:\n'
                        '#   "имя", 8, 1001, 7, центральный_меридиан, 0, 1, '
                        'ложное_смещение_E, ложное_смещение_N\n'
                        '# Строка без кавычек в начале — заголовок раздела, добавляется\n'
                        '# к названиям систем ниже неё. После правки перезапустите программу.\n')
            except OSError as e:
                QMessageBox.warning(self, "Система координат", f"Не удалось создать файл:\n{e}")
                return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(COORD_SYSTEMS_FILE)):
            QMessageBox.warning(self, "Система координат",
                                f"Не удалось открыть файл:\n{COORD_SYSTEMS_FILE}")

    def open_antenna_offset(self):
        dlg = AntennaOffsetDialog(self, self.settings)
        if dlg.exec():
            self.main_window.export_tab.load_geometry_defaults()

    def clear_cache(self):
        profile = QWebEngineProfile.defaultProfile()
        profile.clearHttpCache()
        self.cache_label.setText(f"Текущий размер кэша: {dir_size(profile.cachePath()) / 1e6:.1f} МБ")

    def apply_settings(self):
        self.settings.setValue("depth_mode", "manual" if self.manual_radio.isChecked() else "auto")
        self.settings.setValue("depth_min", self.min_spin.value())
        self.settings.setValue("depth_max", self.max_spin.value())
        self.settings.setValue("depth_steps", self.steps_spin.value())
        self.settings.setValue("hardness_mode",
                               "manual" if self.hardness_manual_radio.isChecked() else "auto")
        self.settings.setValue("hardness_min", self.hardness_min_spin.value())
        self.settings.setValue("hardness_max", self.hardness_max_spin.value())
        self.settings.setValue("hardness_steps", self.hardness_steps_spin.value())
        self.settings.setValue("track_line_width", self.track_width_spin.value())
        self.settings.setValue("speed_glow_width", self.speed_glow_spin.value())
        self.settings.setValue("timeline_axis", "distance" if self.axis_dist_radio.isChecked() else "time")
        cache_mb = self.cache_limit_spin.value()
        self.settings.setValue("cache_max_mb", cache_mb)
        QWebEngineProfile.defaultProfile().setHttpCacheMaximumSize(cache_mb * 1024 * 1024)
        self.main_window.redraw_map()
        self.main_window.sediments_tab.refresh()


class AntennaOffsetDiagram(QWidget):
    """Схема выноса/осадки датчика эхолота относительно ватерлинии и GNSS-
    антенны — иллюстративная (пропорции фиксированы, не масштабируются по
    введённым значениям), поясняет параметры --lever/--draft/--antenna-height
    sl2sync.py (те же поля — в группе «Геометрия» на вкладке «Экспорт
    данных»)."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 320)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))
        w, h = self.width(), self.height()

        water_y = int(h * 0.55)
        mast_x = int(w * 0.35)
        sensor_x = int(w * 0.62)
        antenna_y = int(h * 0.12)
        deck_y = water_y - 18
        sensor_y = int(h * 0.82)

        painter.setPen(QColor("#3388ff"))
        painter.drawLine(20, water_y, w - 20, water_y)
        painter.drawText(24, water_y - 6, "уровень воды")

        hull = [QPoint(mast_x - 70, deck_y), QPoint(mast_x + 90, deck_y),
                QPoint(mast_x + 70, water_y), QPoint(mast_x - 50, water_y)]
        painter.setBrush(QColor("#666666"))
        painter.setPen(QColor("#cccccc"))
        painter.drawPolygon(hull)

        painter.setPen(QColor("#ffffff"))
        painter.drawLine(mast_x, deck_y, mast_x, antenna_y)
        painter.setBrush(QColor("#ffcc00"))
        painter.drawEllipse(QPoint(mast_x, antenna_y), 6, 6)
        painter.drawText(mast_x - 50, antenna_y - 10, "GNSS-антенна")

        painter.setPen(QColor("#ffffff"))
        painter.drawLine(sensor_x, water_y, sensor_x, sensor_y)
        tri = [QPoint(sensor_x - 8, sensor_y), QPoint(sensor_x + 8, sensor_y),
               QPoint(sensor_x, sensor_y + 14)]
        painter.setBrush(QColor("#2ecc71"))
        painter.drawPolygon(tri)
        painter.drawText(sensor_x + 12, sensor_y + 10, "Датчик эхолота")

        painter.setPen(QColor("#ff8800"))
        self._v_dim(painter, mast_x - 100, antenna_y, water_y, "h — высота антенны")
        self._v_dim(painter, sensor_x + 40, water_y, sensor_y, "d — осадка")

        painter.setPen(QColor("#00e5ff"))
        self._h_dim(painter, mast_x, sensor_x, water_y + 25, "L — вынос вперёд")

        self._draw_top_view(painter, w - 160, 16, 130, 100)

    @staticmethod
    def _v_dim(painter, x, y1, y2, label):
        painter.drawLine(x, y1, x, y2)
        painter.drawLine(x - 4, y1, x + 4, y1)
        painter.drawLine(x - 4, y2, x + 4, y2)
        painter.save()
        painter.translate(x - 8, (y1 + y2) / 2 + 40)
        painter.rotate(-90)
        painter.drawText(0, 0, label)
        painter.restore()

    @staticmethod
    def _h_dim(painter, x1, x2, y, label):
        painter.drawLine(x1, y, x2, y)
        painter.drawLine(x1, y - 4, x1, y + 4)
        painter.drawLine(x2, y - 4, x2, y + 4)
        painter.drawText((x1 + x2) // 2 - 45, y + 16, label)

    @staticmethod
    def _draw_top_view(painter, x0, y0, w, h):
        painter.setPen(QColor("#888888"))
        painter.setBrush(QColor("#252525"))
        painter.drawRect(x0, y0, w, h)
        painter.drawText(x0, y0 - 6, "Вид сверху")
        ax, ay = x0 + 20, y0 + h - 20
        sx, sy = x0 + 100, y0 + 25
        painter.setPen(QColor("#00e5ff"))
        painter.drawLine(ax, ay, ax, sy)
        painter.drawLine(ax, sy, sx, sy)
        painter.setBrush(QColor("#ffcc00"))
        painter.drawEllipse(QPoint(ax, ay), 5, 5)
        painter.setBrush(QColor("#2ecc71"))
        painter.drawEllipse(QPoint(sx, sy), 5, 5)
        painter.setPen(QColor("#cccccc"))
        painter.drawText(ax - 18, (ay + sy) // 2, "L")
        painter.drawText((ax + sx) // 2 - 4, sy - 6, "R")


class AntennaOffsetDialog(QDialog):
    """Настройки → «Смещение антенн…» — те же геометрические параметры, что
    в группе «Геометрия» на вкладке «Экспорт данных» (--lever/--draft/
    --antenna-height sl2sync.py), но со схемой-пояснением и сохранением как
    значения по умолчанию (QSettings) — чтобы не вводить их заново на каждый
    расчёт, если геометрия судна/оборудования не меняется между выходами."""

    def __init__(self, parent, settings):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Смещение антенн")

        self.diagram = AntennaOffsetDiagram()

        self.lever_fwd_spin = QDoubleSpinBox()
        self.lever_fwd_spin.setRange(-100.0, 100.0)
        self.lever_fwd_spin.setDecimals(3)
        self.lever_fwd_spin.setSuffix(" м")
        self.lever_fwd_spin.setValue(float(settings.value("geom_lever_fwd", 0.0)))
        self.lever_right_spin = QDoubleSpinBox()
        self.lever_right_spin.setRange(-100.0, 100.0)
        self.lever_right_spin.setDecimals(3)
        self.lever_right_spin.setSuffix(" м")
        self.lever_right_spin.setValue(float(settings.value("geom_lever_right", 0.0)))
        self.draft_spin = QDoubleSpinBox()
        self.draft_spin.setRange(-100.0, 100.0)
        self.draft_spin.setDecimals(3)
        self.draft_spin.setSuffix(" м")
        self.draft_spin.setValue(float(settings.value("geom_draft", 0.0)))
        self.antenna_height_spin = QDoubleSpinBox()
        self.antenna_height_spin.setRange(-100.0, 100.0)
        self.antenna_height_spin.setDecimals(3)
        self.antenna_height_spin.setSuffix(" м")
        self.antenna_height_spin.setValue(float(settings.value("geom_antenna_height", 0.0)))

        form = QFormLayout()
        form.addRow(label_with_help(
            "Вынос вперёд, L:",
            "Расстояние от GNSS-антенны до датчика эхолота вдоль курса судна, "
            "метры. Вперёд от антенны — положительное значение."),
            self.lever_fwd_spin)
        form.addRow(label_with_help(
            "Вынос вправо, R:",
            "Расстояние от GNSS-антенны до датчика эхолота поперёк курса судна, "
            "метры (см. схему «Вид сверху»). Вправо от антенны — положительное "
            "значение."),
            self.lever_right_spin)
        form.addRow(label_with_help(
            "Осадка датчика, d:",
            "Заглубление датчика эхолота под уровень воды, метры (keel offset "
            "в самом Lowrance при этом должен быть выставлен в 0 — иначе "
            "поправка будет учтена дважды)."),
            self.draft_spin)
        form.addRow(label_with_help(
            "Высота антенны, h:",
            "Высота фазового центра GNSS-антенны над уровнем воды, метры — "
            "нужна, только если требуется отметка дна (превышение), а не "
            "только глубина."),
            self.antenna_height_spin)

        note = QLabel(
            "Схема — иллюстративная (пропорции не отражают реальные размеры). "
            "Значения сохраняются как умолчания для группы «Геометрия» на "
            "вкладке «Экспорт данных» — там их всегда можно переопределить "
            "под конкретный расчёт.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #999;")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.save_and_accept)
        buttons.rejected.connect(self.reject)

        content_row = QHBoxLayout()
        content_row.addWidget(self.diagram, 1)
        content_row.addLayout(form)

        layout = QVBoxLayout()
        layout.addLayout(content_row, 1)
        layout.addWidget(note)
        layout.addWidget(buttons)
        self.setLayout(layout)
        self.resize(820, 420)

    def save_and_accept(self):
        self.settings.setValue("geom_lever_fwd", self.lever_fwd_spin.value())
        self.settings.setValue("geom_lever_right", self.lever_right_spin.value())
        self.settings.setValue("geom_draft", self.draft_spin.value())
        self.settings.setValue("geom_antenna_height", self.antenna_height_spin.value())
        self.accept()


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
        row1.addWidget(QLabel("Эхограмма:"))
        row1.addWidget(help_icon(SL2_FILE_HELP))
        row1.addWidget(self.sl2_edit, 1)
        row1.addWidget(sl2_btn)
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
        row2.addWidget(QLabel("Ровер:"))
        row2.addWidget(help_icon(GNSS_FILE_HELP))
        row2.addWidget(self.gnss_edit, 1)
        row2.addWidget(gnss_btn)
        row2.addWidget(sep2a)
        row2.addWidget(multi_gnss_btn)
        row2.addWidget(sep2b)
        row2.addWidget(close_gnss_btn)

        self.gnss_base_path = ""
        self.gnss_base_edit = QLineEdit(readOnly=True)
        gnss_base_btn = QPushButton("Обзор…")
        gnss_base_btn.clicked.connect(self.pick_gnss_base)
        close_gnss_base_btn = QPushButton("Закрыть")
        close_gnss_base_btn.clicked.connect(self.close_gnss_base)
        row2b = QHBoxLayout()
        row2b.addWidget(QLabel("База:"))
        row2b.addWidget(help_icon(GNSS_BASE_FILE_HELP))
        row2b.addWidget(self.gnss_base_edit, 1)
        row2b.addWidget(gnss_base_btn)
        row2b.addWidget(close_gnss_base_btn)

        sep_hline = QFrame()
        sep_hline.setFrameShape(QFrame.Shape.HLine)
        sep_hline.setFrameShadow(QFrame.Shadow.Sunken)

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

        map_tab_layout = QVBoxLayout()
        map_tab_layout.addLayout(maps_row, 1)
        map_tab_layout.addLayout(row3)
        map_tab = QWidget()
        map_tab.setLayout(map_tab_layout)

        self.echogram_tab = EchogramTab(self)
        self.isobaths_tab = IsobathsTab(self)
        self.offset_tab = OffsetTab(self)
        self.ppk_tab = PPKTab(self)
        self.sediments_tab = SedimentsTab(self)
        self.export_tab = ExportTab(self)
        self.help_tab = HelpTab()
        self.settings_tab = SettingsTab(self)

        self.tabs = QTabWidget()
        self.tabs.addTab(map_tab, "Карта")
        self.tabs.addTab(self.echogram_tab, "Эхограмма")
        self.tabs.addTab(self.ppk_tab, "PPK")
        self.tabs.addTab(self.offset_tab, "Расчёт смещения")
        self.tabs.addTab(self.isobaths_tab, "Построение изобат")
        self.tabs.addTab(self.sediments_tab, "Донные отложения")
        self.tabs.addTab(self.export_tab, "Экспорт данных")
        self.tabs.addTab(self.help_tab, "Справка")
        self.tabs.addTab(self.settings_tab, "Настройки")
        self.tabs.currentChanged.connect(self.on_tab_changed)

        layout = QVBoxLayout()
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row2b)
        layout.addWidget(sep_hline)
        layout.addWidget(self.tabs, 1)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Выберите файл эхограммы")
        QTimer.singleShot(500, lambda: self.check_for_updates(silent=True))

    def on_tab_changed(self, index):
        widget = self.tabs.widget(index)
        if widget is self.isobaths_tab:
            self.isobaths_tab.refresh_track()
        elif widget is self.echogram_tab:
            self.echogram_tab.refresh()
        elif widget is self.sediments_tab:
            self.sediments_tab.refresh()

    def pick_sl2(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Файл эхограммы", self.last_dir(), "Lowrance (*.sl2)")
        if path:
            self.remember_dir(path)
            self.file_data = []
            self.depth_offset_per_file = {}
            self.load_file(path, ask_more=False)

    def pick_gnss(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ровер", self.last_dir(), "GNSS-трек (*.nmea *.pos *.csv *.ubx);;Все файлы (*)")
        if path:
            self.remember_dir(path)
            self.gnss_paths = [path]
            self.gnss_edit.setText(path)
            self.update_offset_enabled()
            self.reload_gnss_track()
            if path.lower().endswith(".ubx"):
                self.ppk_tab.add_file_unique(self.ppk_tab.rovers, path)

    def pick_multi_gnss(self):
        self.gnss_paths = []
        self._pick_next_gnss_file()

    def _pick_next_gnss_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ровер", self.last_dir(), "GNSS-трек (*.nmea *.pos *.csv *.ubx);;Все файлы (*)")
        if not path:
            return
        self.remember_dir(path)
        self.gnss_paths.append(path)
        self.gnss_edit.setText("; ".join(self.gnss_paths))
        self.update_offset_enabled()
        self.reload_gnss_track()
        if path.lower().endswith(".ubx"):
            self.ppk_tab.add_file_unique(self.ppk_tab.rovers, path)
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

    def pick_gnss_base(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "GNSS база", self.last_dir(), "UBX (*.ubx);;Все файлы (*)")
        if path:
            self.remember_dir(path)
            self.gnss_base_path = path
            self.gnss_base_edit.setText(path)
            self.ppk_tab.add_file_unique(self.ppk_tab.bases, path)

    def close_gnss_base(self):
        self.gnss_base_path = ""
        self.gnss_base_edit.setText("")

    def set_gnss_track(self, paths):
        """Устанавливает GNSS-трек программно (например, результат PPK) — та же
        логика, что при выборе файла вручную через pick_gnss."""
        self.gnss_paths = list(paths)
        self.gnss_edit.setText("; ".join(self.gnss_paths))
        self.update_offset_enabled()
        self.reload_gnss_track()

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
        self.offset_tab.calc_btn.setEnabled(bool(path) and ";" not in path and bool(self.gnss_paths))

    def apply_offset_dates(self, result):
        # Дата/время из mtime файла .sl2 — лишь приближение (см. docs/CONTEXT.md);
        # синхронизация с GNSS даёт настоящее UTC-время, поэтому раз она уже
        # посчитана (только для одного загруженного файла — расчёт смещения
        # не поддерживает мультиэхограмму), обновляем сводку точным значением.
        if self.points and len(self.file_data) == 1 and result.get("start") and result.get("end"):
            self.points["start"] = result["start"]
            self.points["end"] = result["end"]
            self.points["duration_s"] = result["duration_s"]
            self.update_summary()

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

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(ICON_PATH))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
