"""Generación robusta de curvas de nivel desde un DEM GeoTIFF y exportación a KMZ/KML.

Versión v6.5:
- evita curvas falsas en bordes NoData;
- agrega fallback si falla scikit-image;
- genera vista previa con curvas;
- entrega mensajes de error más claros para Streamlit Cloud.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ContourResult:
    kmz_path: Path
    kml_path: Path
    preview_png: Path | None
    n_lines: int
    min_elevation: float
    max_elevation: float
    interval: float


def _require_geospatial_libs():
    try:
        import rasterio  # noqa: F401
        from rasterio import Affine  # noqa: F401
        from shapely.geometry import LineString  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Faltan librerías geoespaciales. Revisa requirements.txt y packages.txt: "
            "rasterio, shapely, pyproj, GDAL/PROJ/GEOS. simplekml es opcional desde v6.5."
        ) from exc


def _load_dem_limited(dem_path: Path, max_cells: int = 4_000_000):
    import rasterio
    from rasterio import Affine

    if not Path(dem_path).exists():
        raise FileNotFoundError(f"No existe el DEM: {dem_path}")

    with rasterio.open(dem_path) as src:
        data = src.read(1, masked=True).astype("float64")
        nodata = src.nodata
        data = data.filled(np.nan)

        # Protección adicional cuando el archivo no viene en máscara pero sí con nodata.
        if nodata is not None:
            data = np.where(np.isclose(data, nodata), np.nan, data)

        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        factor = 1
        cells = int(data.shape[0] * data.shape[1])

        if cells > max_cells:
            factor = int(math.ceil(math.sqrt(cells / max_cells)))
            data = data[::factor, ::factor]
            transform = transform * Affine.scale(factor, factor)

        finite = np.isfinite(data)
        if finite.sum() < 25:
            raise ValueError("El DEM no contiene suficientes celdas válidas para generar curvas.")

        return data, transform, crs, bounds, factor


def _levels_from_dem(data: np.ndarray, interval: float) -> np.ndarray:
    if interval <= 0:
        raise ValueError("La equidistancia debe ser mayor que cero.")
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("El DEM no contiene datos válidos de elevación.")

    zmin = float(np.nanpercentile(finite, 0.2))
    zmax = float(np.nanpercentile(finite, 99.8))
    if zmax <= zmin:
        zmin = float(np.nanmin(finite))
        zmax = float(np.nanmax(finite))

    start = math.ceil(zmin / interval) * interval
    end = math.floor(zmax / interval) * interval

    if end < start:
        return np.array([round((zmin + zmax) / 2.0, 2)])

    levels = np.arange(start, end + interval, interval, dtype=float)
    # Evitar generar miles de niveles por accidente.
    if len(levels) > 500:
        step = max(1, math.ceil(len(levels) / 500))
        levels = levels[::step]
    return levels


def _maybe_transform_coords(coords, crs):
    if not coords:
        return coords
    if crs is None:
        return coords
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    if epsg == 4326:
        return coords
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xs, ys = zip(*coords)
        lon, lat = transformer.transform(xs, ys)
        return list(zip(lon, lat))
    except Exception:
        return coords


def _valid_contour_segment(arr: np.ndarray, valid_mask: np.ndarray) -> bool:
    """Descarta curvas falsas generadas en bordes NoData."""
    if arr is None or len(arr) < 3:
        return False

    rows = np.clip(np.rint(arr[:, 0]).astype(int), 0, valid_mask.shape[0] - 1)
    cols = np.clip(np.rint(arr[:, 1]).astype(int), 0, valid_mask.shape[1] - 1)
    vals = valid_mask[rows, cols]

    # Debe tener mayoría amplia de puntos válidos. Si pasa sobre NoData, se descarta.
    return float(np.mean(vals)) >= 0.98


def _find_contours_skimage(data: np.ndarray, levels: np.ndarray):
    from skimage import measure

    valid = np.isfinite(data)
    finite = data[valid]
    fill_value = float(np.nanmin(finite) - 10.0 * max(1.0, np.nanstd(finite)))
    safe_data = np.where(valid, data, fill_value)

    for level in levels:
        try:
            contours = measure.find_contours(safe_data, level=float(level))
        except Exception:
            contours = []
        for arr in contours:
            if _valid_contour_segment(arr, valid):
                yield float(level), arr


def _find_contours_matplotlib(data: np.ndarray, levels: np.ndarray):
    """Fallback robusto usando matplotlib.contour."""
    import matplotlib.pyplot as plt

    valid = np.isfinite(data)
    finite = data[valid]
    fill_value = float(np.nanmin(finite) - 10.0 * max(1.0, np.nanstd(finite)))
    safe_data = np.where(valid, data, fill_value)

    fig, ax = plt.subplots()
    try:
        cs = ax.contour(safe_data, levels=levels)
        for level, collection in zip(cs.levels, cs.collections):
            for path in collection.get_paths():
                verts = path.vertices
                if verts is None or len(verts) < 3:
                    continue
                # Matplotlib entrega x=col, y=row. Convertimos a row,col.
                arr = np.column_stack([verts[:, 1], verts[:, 0]])
                if _valid_contour_segment(arr, valid):
                    yield float(level), arr
    finally:
        plt.close(fig)


def _contour_iterator(data: np.ndarray, levels: np.ndarray):
    try:
        yielded = False
        for level, arr in _find_contours_skimage(data, levels):
            yielded = True
            yield level, arr
        if yielded:
            return
    except Exception:
        pass

    # Fallback si skimage no está disponible o no genera curvas válidas.
    yield from _find_contours_matplotlib(data, levels)


def generate_contours_kmz(
    dem_path: Path,
    output_dir: Path,
    interval: float = 50,
    simplify_m: float = 60,
    max_cells: int = 4_000_000,
    max_lines: int = 20000,
    progress_callback=None,
) -> ContourResult:
    """Genera curvas de nivel desde DEM y exporta KML/KMZ.

    El DEM se decima internamente si es muy grande, para evitar caídas en Streamlit Cloud.
    """
    _require_geospatial_libs()

    import matplotlib.pyplot as plt
    from shapely.geometry import LineString
    import zipfile
    from xml.sax.saxutils import escape

    output_dir.mkdir(parents=True, exist_ok=True)
    kmz_path = output_dir / "curvas_nivel.kmz"
    kml_path = output_dir / "curvas_nivel.kml"
    preview_png = output_dir / "preview_curvas.png"

    data, transform, crs, bounds, factor = _load_dem_limited(dem_path, max_cells=max_cells)
    finite = data[np.isfinite(data)]
    zmin = float(np.nanmin(finite))
    zmax = float(np.nanmax(finite))
    levels = _levels_from_dem(data, interval=interval)

    kml_placemarks: list[str] = []
    kml_description = (
        f"Equidistancia: {interval} m | Celdas procesadas: {data.shape[0]} x {data.shape[1]} | "
        f"Factor decimación: {factor}"
    )

    total_levels = max(len(levels), 1)
    n_lines = 0

    if crs is not None and getattr(crs, "is_geographic", False):
        tolerance = simplify_m / 111_000.0
    else:
        tolerance = simplify_m

    current_level_index = 0
    last_level = None

    for level, arr in _contour_iterator(data, levels):
        if last_level != level:
            current_level_index += 1
            last_level = level
            if progress_callback:
                progress_callback(
                    min(current_level_index, total_levels),
                    total_levels,
                    f"Procesando curva {level:g} m",
                )

        coords = []
        for row, col in arr:
            x, y = transform * (float(col), float(row))
            coords.append((x, y))

        coords = _maybe_transform_coords(coords, crs)
        if len(coords) < 3:
            continue

        try:
            line = LineString(coords)
            if simplify_m > 0:
                line = line.simplify(tolerance, preserve_topology=False)
            if line.is_empty or len(line.coords) < 2:
                continue
            line_coords = list(line.coords)
        except Exception:
            line_coords = coords

        coord_text = " ".join(
            f"{float(x):.8f},{float(y):.8f},{float(level):.3f}" for x, y in line_coords
        )
        kml_placemarks.append(
            "<Placemark>"
            f"<name>{escape(f'Curva {level:g} m')}</name>"
            f"<description>{escape(f'Cota {level:g} m')}</description>"
            "<Style><LineStyle><color>ff00aaff</color><width>1.2</width></LineStyle></Style>"
            "<LineString><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode>"
            f"<coordinates>{coord_text}</coordinates>"
            "</LineString></Placemark>"
        )
        n_lines += 1

        if n_lines >= max_lines:
            break

    if n_lines == 0:
        raise RuntimeError(
            "No se generaron curvas de nivel. Prueba con mayor margen de DEM, menor equidistancia "
            "o verifica que el GeoTIFF tenga valores de elevación válidos."
        )

    if progress_callback:
        progress_callback(total_levels, total_levels, "Guardando KML/KMZ")

    kml_text = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        '<Document>\n'
        '<name>Curvas de nivel generadas desde DEM</name>\n'
        f'<description>{escape(kml_description)}</description>\n'
        + "\n".join(kml_placemarks)
        + "\n</Document>\n</kml>\n"
    )
    kml_path.write_text(kml_text, encoding="utf-8")
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(kml_path, arcname="doc.kml")

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(data, cmap="terrain")
        ax.contour(data, levels=levels[: min(len(levels), 40)], linewidths=0.35)
        ax.set_title("Vista previa del DEM y curvas calculadas")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(preview_png, dpi=150)
        plt.close(fig)
    except Exception:
        preview_png = None

    return ContourResult(
        kmz_path=kmz_path,
        kml_path=kml_path,
        preview_png=preview_png,
        n_lines=n_lines,
        min_elevation=zmin,
        max_elevation=zmax,
        interval=interval,
    )
