"""Descarga de DEM COP30 desde OpenTopography.

Incluye modo normal y modo mosaico por teselas, pensado para cuencas grandes.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import requests

OPENTOPO_GLOBALDEM_URL = "https://portal.opentopography.org/API/globaldem"


@dataclass
class BoundingBox:
    south: float
    north: float
    west: float
    east: float

    def validate(self) -> None:
        if self.south >= self.north:
            raise ValueError("El límite sur debe ser menor que el límite norte.")
        if self.west >= self.east:
            raise ValueError("El límite oeste debe ser menor que el límite este.")
        if not (-90 <= self.south <= 90 and -90 <= self.north <= 90):
            raise ValueError("Latitudes fuera de rango.")
        if not (-180 <= self.west <= 180 and -180 <= self.east <= 180):
            raise ValueError("Longitudes fuera de rango.")

    def to_params(self) -> dict[str, float]:
        return asdict(self)


def bbox_from_point(lat: float, lon: float, margin_degrees: float) -> BoundingBox:
    bbox = BoundingBox(
        south=max(-90, lat - margin_degrees),
        north=min(90, lat + margin_degrees),
        west=max(-180, lon - margin_degrees),
        east=min(180, lon + margin_degrees),
    )
    bbox.validate()
    return bbox


def bbox_from_point_km(lat: float, lon: float, margin_km: float) -> BoundingBox:
    # Conversión suficiente para definir el rectángulo inicial; la descarga sigue en coordenadas geográficas.
    deg_lat = margin_km / 111.32
    cos_lat = max(0.15, math.cos(math.radians(lat)))
    deg_lon = margin_km / (111.32 * cos_lat)
    bbox = BoundingBox(
        south=max(-90, lat - deg_lat),
        north=min(90, lat + deg_lat),
        west=max(-180, lon - deg_lon),
        east=min(180, lon + deg_lon),
    )
    bbox.validate()
    return bbox


def split_bbox(bbox: BoundingBox, n_tiles: int) -> list[BoundingBox]:
    n_tiles = max(1, int(n_tiles))
    cols = math.ceil(math.sqrt(n_tiles))
    rows = math.ceil(n_tiles / cols)
    dlon = (bbox.east - bbox.west) / cols
    dlat = (bbox.north - bbox.south) / rows
    tiles: list[BoundingBox] = []
    for r in range(rows):
        for c in range(cols):
            if len(tiles) >= n_tiles:
                break
            west = bbox.west + c * dlon
            east = bbox.west + (c + 1) * dlon
            south = bbox.south + r * dlat
            north = bbox.south + (r + 1) * dlat
            tiles.append(BoundingBox(south=south, north=north, west=west, east=east))
    return tiles


def _check_response_for_error(response: requests.Response) -> None:
    content_type = response.headers.get("content-type", "").lower()
    if "text" in content_type or "json" in content_type or "html" in content_type:
        text = response.text[:1000]
        if "error" in text.lower() or response.status_code >= 400:
            raise RuntimeError(f"OpenTopography devolvió un error: {text}")


def download_dem_cop30(
    bbox: BoundingBox,
    api_key: str,
    output_path: Path,
    demtype: str = "COP30",
    timeout: int = 600,
) -> Path:
    """Descarga un DEM GeoTIFF para un bbox."""
    bbox.validate()
    if not api_key:
        raise ValueError("Falta la API key de OpenTopography.")

    params = {
        "demtype": demtype,
        "south": bbox.south,
        "north": bbox.north,
        "west": bbox.west,
        "east": bbox.east,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(OPENTOPO_GLOBALDEM_URL, params=params, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        _check_response_for_error(response)
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if output_path.stat().st_size < 1024:
        raise RuntimeError("El archivo descargado es demasiado pequeño para ser un GeoTIFF válido.")
    return output_path


def merge_geotiffs(tile_paths: Iterable[Path], output_path: Path) -> Path:
    """Une varios GeoTIFF en un solo mosaico."""
    try:
        import rasterio
        from rasterio.merge import merge
    except Exception as exc:
        raise RuntimeError(
            "Para unir DEM parciales se requiere rasterio. Instala rasterio o usa descarga única."
        ) from exc

    sources = []
    try:
        for path in tile_paths:
            sources.append(rasterio.open(path))
        if not sources:
            raise ValueError("No hay teselas para unir.")
        mosaic, out_transform = merge(sources)
        meta = sources[0].meta.copy()
        meta.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "driver": "GTiff",
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **meta) as dest:
            dest.write(mosaic)
        return output_path
    finally:
        for src in sources:
            src.close()


def download_dem_cop30_tiled(
    bbox: BoundingBox,
    api_key: str,
    output_path: Path,
    tiles_dir: Path,
    n_tiles: int = 10,
    demtype: str = "COP30",
    progress_callback=None,
) -> tuple[Path, list[Path]]:
    """Descarga el DEM en teselas y lo une como mosaico."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    tile_bboxes = split_bbox(bbox, n_tiles=n_tiles)
    tile_paths: list[Path] = []
    total = len(tile_bboxes)
    for idx, tile_bbox in enumerate(tile_bboxes, start=1):
        tile_path = tiles_dir / f"dem_tile_{idx:03d}.tif"
        if progress_callback:
            progress_callback(idx - 1, total, f"Descargando DEM parcial {idx}/{total}")
        download_dem_cop30(tile_bbox, api_key=api_key, output_path=tile_path, demtype=demtype)
        tile_paths.append(tile_path)
    if progress_callback:
        progress_callback(total, total, "Uniendo DEM parciales")
    merged = merge_geotiffs(tile_paths, output_path=output_path)
    return merged, tile_paths


def validate_geotiff(path: Path) -> dict:
    """Devuelve metadatos básicos de un GeoTIFF."""
    try:
        import rasterio
    except Exception as exc:
        raise RuntimeError("No se pudo validar el GeoTIFF porque rasterio no está instalado.") from exc

    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": str(src.crs),
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "nodata": src.nodata,
            "dtype": src.dtypes[0] if src.dtypes else None,
        }
