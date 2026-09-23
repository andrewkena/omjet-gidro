# sl2sync — привязка глубин Lowrance .sl2 к точному GNSS

## Установка
    pip install -r requirements.txt

## Запуск
    python sl2sync.py запись.sl2 трек.nmea --lever 0.5 0.3 --draft 0.1 --antenna-height 1.2 --auto-latency

GNSS-трек: NMEA-лог (GGA + RMC/ZDA), RTKLIB .pos (PPK) или CSV с колонками time, lat, lon[, h, q].

## Основные параметры
- `--lever ВПЕРЁД ВПРАВО` — где датчик эхолота относительно GNSS-антенны, м.
- `--draft` — заглубление датчика под водой, м (keel offset в Lowrance поставьте 0).
- `--antenna-height` — высота антенны над водой, м; включает расчёт отметок дна (нужен RTK/PPK).
- `--auto-latency` — подбор остаточной задержки глубины по встречным/пересекающимся галсам.
- `--latency` — задать задержку вручную (если уже измерена).
- `--offset`, `--drift-ppm` — ручная синхронизация, если авто не сработала.
- `--min-fix fix` — брать только точки с RTK FIX.

## Результат (папка <имя>_out)
- `*_points.csv` — точки: время UTC, lat/lon, UTM, глубина, отметка дна, качество GNSS, курс.
- `*_depth.asc` / `*_zbottom.asc` + `.prj` — грид для QGIS (GeoTIFF, если установлен rasterio).
- `*_report.png` — графики синхронизации, подбора задержки и карта точек.
- `*_summary.json` — найденные смещение, дрейф, задержка и статистика.

Пример отчёта:

![report](docs/example_report.png)

## Проверка на синтетике
    python make_test_data.py
    python sl2sync.py test.sl2 test_gnss.nmea --lever 0.5 0.3 --draft 0.1 --antenna-height 1.0 --auto-latency
