
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import heapq
import math
import zipfile
from collections import deque
from xml.sax.saxutils import escape

import numpy as np


@dataclass
class WatershedResult:
    kmz_path: Path
    kml_path: Path
    preview_png: Path | None
    area_km2: float
    perimeter_km: float
    outlet_original_lon: float
    outlet_original_lat: float
    outlet_snapped_lon: float
    outlet_snapped_lat: float
    snapped_distance_m: float
    n_cells_basin: int
    cell_size_m: float
    accumulation_at_outlet_cells: float
    quality_flags: list[str]


def _cell_sizes_m(transform, crs, data_shape) -> tuple[float, float, float]:
    dx_raw = abs(float(transform.a))
    dy_raw = abs(float(transform.e))
    try:
        is_geographic = bool(crs and getattr(crs, "is_geographic", False))
    except Exception:
        is_geographic = False

    if is_geographic:
        rows = data_shape[0]
        y_mid = transform.f + transform.e * (rows / 2.0)
        dx = dx_raw * 111_320.0 * max(0.15, math.cos(math.radians(float(y_mid))))
        dy = dy_raw * 110_574.0
    else:
        dx, dy = dx_raw, dy_raw

    return float(dx), float(dy), float(math.sqrt(dx * dy))


def _read_dem_for_watershed(dem_path: Path, max_cells: int = 1_500_000):
    import rasterio
    from rasterio import Affine

    if not Path(dem_path).exists():
        raise FileNotFoundError(f"No existe el DEM: {dem_path}")

    with rasterio.open(dem_path) as src:
        data = src.read(1, masked=True).astype("float64").filled(np.nan)
        if src.nodata is not None:
            data = np.where(np.isclose(data, src.nodata), np.nan, data)

        transform = src.transform
        crs = src.crs
        factor = 1
        cells = int(data.shape[0] * data.shape[1])
        if cells > max_cells:
            factor = int(math.ceil(math.sqrt(cells / max_cells)))
            data = data[::factor, ::factor]
            transform = transform * Affine.scale(factor, factor)

    finite = np.isfinite(data)
    if int(finite.sum()) < 100:
        raise ValueError("El DEM no tiene suficientes celdas válidas para delimitar cuenca.")
    return data, transform, crs, factor


def _lonlat_to_rowcol(lon: float, lat: float, transform, crs):
    if crs is not None:
        try:
            epsg = crs.to_epsg()
        except Exception:
            epsg = None
        if epsg != 4326:
            try:
                from pyproj import Transformer
                tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                lon, lat = tr.transform(lon, lat)
            except Exception:
                pass
    inv = ~transform
    col, row = inv * (float(lon), float(lat))
    return int(round(row)), int(round(col))


def _rowcol_to_lonlat(row: float, col: float, transform, crs):
    x, y = transform * (float(col), float(row))
    if crs is not None:
        try:
            epsg = crs.to_epsg()
        except Exception:
            epsg = None
        if epsg != 4326:
            try:
                from pyproj import Transformer
                tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                x, y = tr.transform(x, y)
            except Exception:
                pass
    return float(x), float(y)


def _priority_flood_fill(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    nrows, ncols = dem.shape
    filled = dem.copy()
    visited = np.zeros_like(valid, dtype=bool)
    heap: list[tuple[float, int, int]] = []

    for r in range(nrows):
        for c in (0, ncols - 1):
            if valid[r, c] and not visited[r, c]:
                visited[r, c] = True
                heapq.heappush(heap, (filled[r, c], r, c))
    for c in range(ncols):
        for r in (0, nrows - 1):
            if valid[r, c] and not visited[r, c]:
                visited[r, c] = True
                heapq.heappush(heap, (filled[r, c], r, c))

    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc in neigh:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                continue
            if not valid[rr, cc] or visited[rr, cc]:
                continue
            visited[rr, cc] = True
            if filled[rr, cc] < z:
                filled[rr, cc] = z
            heapq.heappush(heap, (filled[rr, cc], rr, cc))
    return filled


def _flow_direction_d8(filled: np.ndarray, valid: np.ndarray, dx: float, dy: float) -> np.ndarray:
    nrows, ncols = filled.shape
    dst = np.full(nrows * ncols, -1, dtype=np.int64)
    neigh = [
        (-1, -1, math.hypot(dx, dy)), (-1, 0, dy), (-1, 1, math.hypot(dx, dy)),
        (0, -1, dx),                         (0, 1, dx),
        (1, -1, math.hypot(dx, dy)),  (1, 0, dy),  (1, 1, math.hypot(dx, dy)),
    ]
    for r in range(nrows):
        base = r * ncols
        for c in range(ncols):
            if not valid[r, c]:
                continue
            z = filled[r, c]
            best_slope = 0.0
            best = -1
            for dr, dc, dist in neigh:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols or not valid[rr, cc]:
                    continue
                slope = (z - filled[rr, cc]) / max(dist, 1e-9)
                if slope > best_slope:
                    best_slope = slope
                    best = rr * ncols + cc
            dst[base + c] = best
    return dst


def _flow_accumulation(dst: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n = dst.size
    valid_flat = valid.ravel()
    indeg = np.zeros(n, dtype=np.int32)
    edges = np.where((dst >= 0) & valid_flat)[0]
    np.add.at(indeg, dst[edges], 1)
    acc = np.zeros(n, dtype=np.float64)
    acc[valid_flat] = 1.0
    q = deque(np.where(valid_flat & (indeg == 0))[0].tolist())
    while q:
        i = q.popleft()
        j = int(dst[i])
        if j >= 0:
            acc[j] += acc[i]
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(j)
    return acc.reshape(valid.shape)


def _snap_outlet(row: int, col: int, acc: np.ndarray, valid: np.ndarray, radius_cells: int):
    nrows, ncols = acc.shape
    row = int(np.clip(row, 0, nrows - 1))
    col = int(np.clip(col, 0, ncols - 1))
    r0 = max(0, row - radius_cells)
    r1 = min(nrows, row + radius_cells + 1)
    c0 = max(0, col - radius_cells)
    c1 = min(ncols, col + radius_cells + 1)
    sub_acc = acc[r0:r1, c0:c1].copy()
    sub_valid = valid[r0:r1, c0:c1]
    yy, xx = np.indices(sub_acc.shape)
    dist2 = (yy + r0 - row) ** 2 + (xx + c0 - col) ** 2
    sub_acc[~sub_valid] = -1
    score = sub_acc - 1e-6 * dist2
    local = np.unravel_index(int(np.nanargmax(score)), score.shape)
    return int(local[0] + r0), int(local[1] + c0)


def _upstream_mask(dst: np.ndarray, valid: np.ndarray, outlet_idx: int) -> np.ndarray:
    n = dst.size
    valid_flat = valid.ravel()
    edges_src = np.where((dst >= 0) & valid_flat)[0]
    edges_dst = dst[edges_src]
    order = np.argsort(edges_dst, kind="mergesort")
    sorted_dst = edges_dst[order]
    sorted_src = edges_src[order]
    basin_flat = np.zeros(n, dtype=bool)
    if outlet_idx < 0 or outlet_idx >= n or not valid_flat[outlet_idx]:
        return basin_flat.reshape(valid.shape)
    stack = [int(outlet_idx)]
    basin_flat[outlet_idx] = True
    while stack:
        target = stack.pop()
        lo = np.searchsorted(sorted_dst, target, side="left")
        hi = np.searchsorted(sorted_dst, target, side="right")
        for child in sorted_src[lo:hi]:
            child = int(child)
            if not basin_flat[child]:
                basin_flat[child] = True
                stack.append(child)
    return basin_flat.reshape(valid.shape)


def _basin_polygon_from_mask(mask: np.ndarray, transform, crs, simplify_m: float = 50.0):
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import transform as shp_transform
    from skimage import measure
    # Se rellena con borde falso para cerrar cuencas que tocan el límite del DEM.
    padded = np.pad(mask.astype(float), 1, mode="constant", constant_values=0.0)
    contours = measure.find_contours(padded, 0.5)
    polys = []
    for arr in contours:
        if len(arr) < 4:
            continue
        coords_xy = []
        for row, col in arr:
            # Restar el padding de 1 celda.
            row = float(row) - 1.0
            col = float(col) - 1.0
            x, y = transform * (float(col), float(row))
            coords_xy.append((x, y))
        if coords_xy[0] != coords_xy[-1]:
            coords_xy.append(coords_xy[0])
        try:
            poly = Polygon(coords_xy)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            if simplify_m > 0:
                try:
                    is_geo = bool(crs and getattr(crs, "is_geographic", False))
                except Exception:
                    is_geo = False
                tol = simplify_m / 111_000.0 if is_geo else simplify_m
                poly = poly.simplify(tol, preserve_topology=True)
            polys.append(poly)
        except Exception:
            continue
    if not polys:
        raise RuntimeError("No se pudo convertir la máscara de cuenca a polígono.")
    poly = max(polys, key=lambda pp: pp.area)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda pp: pp.area)
    if crs is not None:
        try:
            epsg = crs.to_epsg()
        except Exception:
            epsg = None
        if epsg != 4326:
            from pyproj import Transformer
            tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            poly = shp_transform(lambda x, y, z=None: tr.transform(x, y), poly)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _utm_crs_from_lonlat(lon: float, lat: float):
    from pyproj import CRS
    zone = int((lon + 180) // 6) + 1
    epsg = 32700 + zone if lat < 0 else 32600 + zone
    return CRS.from_epsg(epsg)


def _area_perimeter_km(poly_wgs84) -> tuple[float, float]:
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer
    c = poly_wgs84.centroid
    utm = _utm_crs_from_lonlat(float(c.x), float(c.y))
    tr = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    p_m = shp_transform(lambda x, y, z=None: tr.transform(x, y), poly_wgs84)
    return float(p_m.area / 1_000_000.0), float(p_m.length / 1000.0)


def _write_basin_kmz(poly_wgs84, original_lon, original_lat, snapped_lon, snapped_lat, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    kml_path = output_dir / "cuenca_delimitada_automatica.kml"
    kmz_path = output_dir / "cuenca_delimitada_automatica.kmz"
    coords = " ".join([f"{x:.8f},{y:.8f},0" for x, y in list(poly_wgs84.exterior.coords)])
    kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>{escape(name)}</name>
<Style id="basin_style">
  <LineStyle><color>ff0000ff</color><width>2</width></LineStyle>
  <PolyStyle><color>330000ff</color></PolyStyle>
</Style>
<Style id="outlet_original"><IconStyle><scale>1.1</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon></IconStyle></Style>
<Style id="outlet_snapped"><IconStyle><scale>1.1</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/grn-circle.png</href></Icon></IconStyle></Style>
<Placemark>
  <name>Cuenca delimitada automática</name>
  <styleUrl>#basin_style</styleUrl>
  <Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon>
</Placemark>
<Placemark><name>Punto de control original</name><styleUrl>#outlet_original</styleUrl><Point><coordinates>{original_lon:.8f},{original_lat:.8f},0</coordinates></Point></Placemark>
<Placemark><name>Punto ajustado al cauce</name><styleUrl>#outlet_snapped</styleUrl><Point><coordinates>{snapped_lon:.8f},{snapped_lat:.8f},0</coordinates></Point></Placemark>
</Document>
</kml>
'''
    kml_path.write_text(kml, encoding="utf-8")
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(kml_path, arcname="doc.kml")
    return kml_path, kmz_path


def _preview(mask: np.ndarray, acc: np.ndarray, outlet_rc, output_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "preview_cuenca_delimitada.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        acc_log = np.log10(np.where(acc > 0, acc, np.nan))
        ax.imshow(acc_log, cmap="gray")
        ax.contour(mask.astype(float), levels=[0.5], linewidths=1.5)
        ax.scatter([outlet_rc[1]], [outlet_rc[0]], s=30)
        ax.set_title("Cuenca delimitada y red de acumulación")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    except Exception:
        return None


def delineate_watershed_from_dem(
    dem_path: Path,
    outlet_lon: float,
    outlet_lat: float,
    output_dir: Path,
    snap_radius_m: float = 1500.0,
    max_cells: int = 1_500_000,
    simplify_m: float = 80.0,
    project_name: str = "Cuenca HidroSed",
    progress_callback=None,
) -> WatershedResult:
    if progress_callback:
        progress_callback(5, 100, "Leyendo DEM")
    data, transform, crs, decimation_factor = _read_dem_for_watershed(Path(dem_path), max_cells=max_cells)
    valid = np.isfinite(data)
    dx, dy, cell_m = _cell_sizes_m(transform, crs, data.shape)
    if progress_callback:
        progress_callback(15, 100, "Rellenando depresiones del DEM")
    filled = _priority_flood_fill(data, valid)
    if progress_callback:
        progress_callback(35, 100, "Calculando dirección de flujo D8")
    dst = _flow_direction_d8(filled, valid, dx, dy)
    if progress_callback:
        progress_callback(55, 100, "Calculando acumulación de flujo")
    acc = _flow_accumulation(dst, valid)
    row0, col0 = _lonlat_to_rowcol(outlet_lon, outlet_lat, transform, crs)
    if not (0 <= row0 < data.shape[0] and 0 <= col0 < data.shape[1]):
        raise ValueError("El punto de control queda fuera del DEM descargado. Aumenta el margen del DEM.")
    radius_cells = max(1, int(math.ceil(float(snap_radius_m) / max(cell_m, 1e-6))))
    row1, col1 = _snap_outlet(row0, col0, acc, valid, radius_cells)
    outlet_idx = row1 * data.shape[1] + col1
    snapped_lon, snapped_lat = _rowcol_to_lonlat(row1, col1, transform, crs)
    original_lon, original_lat = float(outlet_lon), float(outlet_lat)
    snapped_distance_m = math.hypot((row1 - row0) * dy, (col1 - col0) * dx)
    if progress_callback:
        progress_callback(75, 100, "Delineando área aportante")
    basin = _upstream_mask(dst, valid, outlet_idx)
    n_cells = int(basin.sum())
    flags: list[str] = []
    if n_cells < 50:
        flags.append("Cuenca muy pequeña: revise ubicación del punto de control o radio de ajuste.")
    if snapped_distance_m > max(500.0, 0.5 * snap_radius_m):
        flags.append("El punto de control fue ajustado una distancia importante; revise visualmente el punto de salida.")
    if float(acc[row1, col1]) < 100:
        flags.append("Baja acumulación en el punto de salida; puede no coincidir con el cauce principal.")
    if decimation_factor > 1:
        flags.append(f"DEM decimado por factor {decimation_factor}; para mayor precisión reduzca margen o aumente max_cells.")
    if progress_callback:
        progress_callback(88, 100, "Vectorizando polígono de cuenca")
    poly = _basin_polygon_from_mask(basin, transform, crs, simplify_m=simplify_m)
    area_km2, perim_km = _area_perimeter_km(poly)
    if area_km2 <= 0:
        flags.append("Área calculada no positiva; revisar DEM y punto de salida.")
    if area_km2 < 0.01:
        flags.append("Área menor a 0,01 km²; probablemente no se delimitó la cuenca correcta.")
    if progress_callback:
        progress_callback(95, 100, "Generando KMZ")
    kml_path, kmz_path = _write_basin_kmz(poly, original_lon, original_lat, snapped_lon, snapped_lat, output_dir, project_name)
    preview_png = _preview(basin, acc, (row1, col1), output_dir)
    if progress_callback:
        progress_callback(100, 100, "Cuenca delimitada")
    return WatershedResult(
        kmz_path=kmz_path,
        kml_path=kml_path,
        preview_png=preview_png,
        area_km2=area_km2,
        perimeter_km=perim_km,
        outlet_original_lon=original_lon,
        outlet_original_lat=original_lat,
        outlet_snapped_lon=snapped_lon,
        outlet_snapped_lat=snapped_lat,
        snapped_distance_m=float(snapped_distance_m),
        n_cells_basin=n_cells,
        cell_size_m=float(cell_m),
        accumulation_at_outlet_cells=float(acc[row1, col1]),
        quality_flags=flags,
    )
