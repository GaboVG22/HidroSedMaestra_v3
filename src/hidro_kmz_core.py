"""
Nucleo GIS-hidrologico para aplicacion KMZ -> morfometria -> metodologias hidrologicas.
Disenado para operar con KMZ/KML que contengan poligono de cuenca, curvas de nivel 3D,
eje de cauce y punto de descarga, o con estos ultimos dibujados/cargados por separado.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import io
import math
import os
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, LineString, Point, shape, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from pyproj import CRS, Transformer


# -------------------------------
# Data classes
# -------------------------------
@dataclass
class Feature:
    ftype: str
    name: str
    description: str
    geometry: BaseGeometry
    z_values: List[float]
    properties: Dict[str, Any]

@dataclass
class BasinMetrics:
    area_km2: float
    area_ha: float
    perimeter_km: float
    centroid_lon: float
    centroid_lat: float
    epsg_utm: int
    max_geom_length_km: float
    bbox_length_km: float
    bbox_width_km: float
    mean_width_km: float
    compactness_kc: float
    form_factor: float
    elongation_ratio: float

@dataclass
class ContourMetrics:
    n_contours: int
    n_levels: int
    z_min: Optional[float]
    z_max: Optional[float]
    equidistance_m: Optional[float]
    n_vertices: int

@dataclass
class HydrologyInputs:
    objective: str
    return_periods: List[float]
    land_use: str
    C: float
    CN: float
    intensity_by_T: Dict[float, float]
    p24_by_T: Dict[float, float]
    storm_duration_h: float
    dga_ac_params: Dict[str, Any]
    verni_king_params: Dict[str, Any]


# -------------------------------
# KML/KMZ parsing
# -------------------------------

def _read_uploaded_bytes(file_input: Any) -> Tuple[bytes, str]:
    """Read a path, bytes object or Streamlit UploadedFile safely.

    UploadedFile objects keep an internal pointer. Resetting it avoids intermittent
    errors when Streamlit reruns the app after a widget change.
    """
    if isinstance(file_input, (str, os.PathLike)):
        with open(file_input, "rb") as f:
            return f.read(), str(file_input).lower()
    if isinstance(file_input, bytes):
        return file_input, "uploaded"
    try:
        file_input.seek(0)
    except Exception:
        pass
    data = file_input.read()
    try:
        file_input.seek(0)
    except Exception:
        pass
    return data, getattr(file_input, "name", "uploaded").lower()


def read_kmz_or_kml(file_input: Any) -> str:
    """Return KML text from path, bytes, BytesIO, KMZ or KML.

    Some KMZ files contain several KML files or a small doc.kml with only a
    NetworkLink. The routine now chooses the internal KML with the greatest
    number of Placemarks/coordinate blocks, instead of blindly taking doc.kml.
    """
    data, fname = _read_uploaded_bytes(file_input)

    if zipfile.is_zipfile(io.BytesIO(data)) or fname.endswith(".kmz"):
        try:
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
                if not kml_names:
                    raise ValueError("El KMZ no contiene archivo .kml interno.")
                candidates = []
                for n in kml_names:
                    try:
                        txt = zf.read(n).decode("utf-8", errors="replace")
                    except Exception:
                        continue
                    score = 5 * txt.lower().count("<placemark") + txt.lower().count("<coordinates") + txt.lower().count("<gx:coord")
                    # Prefer doc.kml only if it actually contains geometry.
                    if os.path.basename(n).lower() == "doc.kml":
                        score += 1
                    candidates.append((score, n, txt))
                if not candidates:
                    raise ValueError("El KMZ contiene KML, pero no fue posible leerlos.")
                candidates.sort(key=lambda x: x[0], reverse=True)
                return candidates[0][2]
        except zipfile.BadZipFile:
            raise ValueError("El archivo tiene extensión KMZ, pero no es un ZIP/KMZ válido.")
    return data.decode("utf-8", errors="replace")

def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1]


def _text_or_empty(node: Optional[ET.Element]) -> str:
    return "" if node is None or node.text is None else node.text.strip()


def _sanitize_kml_xml(kml_text: str) -> str:
    """Clean KML/XML text before ElementTree parsing.

    Some KML/KMZ exports from GIS/Google Earth include namespace-prefixed
    attributes (for example xsi:schemaLocation or gx:* tags) without declaring
    the prefix in the root element. ElementTree then raises "unbound prefix".
    This helper adds missing namespace declarations and removes a few illegal
    XML control characters without changing coordinates or attributes.
    """
    if kml_text is None:
        return ""
    text = str(kml_text)
    text = text.lstrip("\ufeff\ufffe\x00\r\n\t ")
    # Remove binary/control characters not allowed in XML 1.0, preserving tabs/newlines.
    text = "".join(ch for ch in text if ch in "\t\n\r" or ord(ch) >= 32)

    # If a server prepends warnings or text before the KML, trim to the first XML/KML tag.
    xml_pos = text.find("<?xml")
    kml_pos = text.lower().find("<kml")
    if xml_pos > 0 and (kml_pos == -1 or xml_pos < kml_pos):
        text = text[xml_pos:]
    elif xml_pos == -1 and kml_pos > 0:
        text = text[kml_pos:]

    # Find first real element start tag, skipping xml declaration/comments/doctypes.
    m = re.search(r"<(?!\?|!)([A-Za-z_][\w.\-]*(?::[A-Za-z_][\w.\-]*)?)([^<>]*?)>", text, flags=re.S)
    if not m:
        return text
    start, end = m.span()
    root_tag = text[start:end]

    declared = set(re.findall(r"\sxmlns:([A-Za-z_][\w.\-]*)\s*=", root_tag))
    # Prefixes used in element names and attribute names.
    used = set(re.findall(r"<\/?([A-Za-z_][\w.\-]*):[A-Za-z_][\w.\-]*", text))
    used.update(re.findall(r"\s([A-Za-z_][\w.\-]*):[A-Za-z_][\w.\-]*\s*=", text))
    used.discard("xml")
    used.discard("xmlns")

    missing = sorted(used - declared)
    if missing:
        ns_map = {
            "gx": "http://www.google.com/kml/ext/2.2",
            "xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "atom": "http://www.w3.org/2005/Atom",
            "xlink": "http://www.w3.org/1999/xlink",
            "kml": "http://www.opengis.net/kml/2.2",
            "fo": "http://www.w3.org/1999/XSL/Format",
            "msxsl": "urn:schemas-microsoft-com:xslt",
        }
        additions = "".join(f' xmlns:{pr}="{ns_map.get(pr, "urn:auto-prefix:" + pr)}"' for pr in missing)
        # Insert declarations before the closing angle bracket of the root tag.
        new_root_tag = root_tag[:-1] + additions + ">"
        text = text[:start] + new_root_tag + text[end:]
    return text


def _strip_xml_prefixes(kml_text: str) -> str:
    """Last-resort fallback: remove XML namespace prefixes from tags/attributes.

    Used only after namespace repair fails. It is intentionally conservative and
    keeps local tag names, so KML geometries such as gx:Track become Track.
    """
    text = str(kml_text)
    text = re.sub(r"<(/?)([A-Za-z_][\w.\-]*):", r"<\1", text)
    text = re.sub(r"\s([A-Za-z_][\w.\-]*):([A-Za-z_][\w.\-]*)\s*=", r" \2=", text)
    return text


def _find_first(parent: ET.Element, tag: str) -> Optional[ET.Element]:
    for child in parent.iter():
        if _strip_ns(child.tag) == tag:
            return child
    return None


def _find_all(parent: ET.Element, tag: str) -> List[ET.Element]:
    return [child for child in parent.iter() if _strip_ns(child.tag) == tag]


def parse_coordinates(coord_text: str) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Parse KML coordinates as lon,lat[,z].

    Robust against line breaks, repeated spaces and altitude strings. Coordinates
    with impossible longitude/latitude are ignored, which prevents a bad vertex
    from killing the entire KMZ import.
    """
    coords: List[Tuple[float, float]] = []
    zs: List[float] = []
    if not coord_text:
        return coords, zs
    text = coord_text.replace("\n", " ").replace("\t", " ").replace(";", " ")
    for token in text.split():
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0].strip())
                lat = float(parts[1].strip())
                z = float(parts[2].strip()) if len(parts) >= 3 and parts[2].strip() != "" else 0.0
                if -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0:
                    coords.append((lon, lat)); zs.append(z)
            except Exception:
                continue
    return coords, zs


def parse_gx_coords(track_el: ET.Element) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Parse gx:coord elements from KML tracks, when present."""
    coords: List[Tuple[float, float]] = []
    zs: List[float] = []
    for child in track_el.iter():
        if _strip_ns(child.tag) == "coord" and child.text:
            parts = child.text.strip().split()
            if len(parts) >= 2:
                try:
                    lon = float(parts[0]); lat = float(parts[1]); z = float(parts[2]) if len(parts) >= 3 else 0.0
                    if -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0:
                        coords.append((lon, lat)); zs.append(z)
                except Exception:
                    continue
    return coords, zs

def _extract_extended_data(pm: ET.Element) -> Dict[str, Any]:
    props = {}
    for data in _find_all(pm, "Data"):
        key = data.attrib.get("name", "")
        val_node = _find_first(data, "value")
        if key:
            props[key] = _text_or_empty(val_node)
    # SimpleData
    for data in _find_all(pm, "SimpleData"):
        key = data.attrib.get("name", "")
        if key:
            props[key] = _text_or_empty(data)
    return props


def elevation_from_text(name: str, description: str, props: Dict[str, Any]) -> Optional[float]:
    """Try to infer elevation from name/description/properties."""
    candidate_texts = [name or "", description or ""] + [str(v) for v in props.values()]
    # Prefer fields with common names
    for k, v in props.items():
        if re.search(r"cota|elev|z|altura|nivel", str(k), flags=re.I):
            m = re.search(r"-?\d+(?:[\.,]\d+)?", str(v))
            if m:
                return float(m.group(0).replace(",", "."))
    for txt in candidate_texts:
        # Examples: CN_500, Cota 520, Elev_1000, curva=560
        patterns = [
            r"(?:CN|COTA|ELEV|ALT|Z|NIVEL)[_\s:=\-]*(-?\d+(?:[\.,]\d+)?)",
            r"(-?\d+(?:[\.,]\d+)?)\s*(?:m|msnm|m\.s\.n\.m\.)",
        ]
        for pat in patterns:
            m = re.search(pat, txt, flags=re.I)
            if m:
                return float(m.group(1).replace(",", "."))
    return None




def _numbers_from_text(txt: str) -> List[float]:
    vals = []
    for m in re.finditer(r"-?\d+(?:[\.,]\d+)?", txt or ""):
        try:
            vals.append(float(m.group(0).replace(",", ".")))
        except Exception:
            pass
    return vals


def _plausible_precip(v: float) -> bool:
    # Daily design precipitation/isohyet values in Chile normally fall in this wide range.
    return np.isfinite(v) and 1.0 <= float(v) <= 1000.0


def _html_table_value(description: str, keys=("VALOR_MM", "P_MM", "P10D", "P24_10", "PRECIPITACION", "PRECIPITACIÓN")) -> Optional[float]:
    """Extract numeric values from simple ArcGIS/Google Earth HTML tables.

    DGA isohyet KMZ exports often store the value in the description as:
    <td>VALOR_MM</td><td>90</td>. This helper strips tags and searches the
    key followed by the next plausible number, avoiding FID/fecha/id numbers.
    """
    if not description:
        return None
    text = re.sub(r"<[^>]+>", " ", description)
    text = re.sub(r"\s+", " ", text)
    for key in keys:
        m = re.search(rf"{re.escape(key)}\s+(-?\d+(?:[\.,]\d+)?)", text, flags=re.I)
        if m:
            try:
                val = float(m.group(1).replace(",", "."))
                if _plausible_precip(val):
                    return val
            except Exception:
                pass
    return None


def precipitation_from_text(name: str, description: str, props: Dict[str, Any]) -> Optional[float]:
    """Infer a precipitation/isohyet value in mm from KML text/properties.

    The v1.7 parser could mistake the return period in strings like
    'Isoyeta P10D 60 mm' for the precipitation value. This version prefers:
    1) explicit attributes (p_mm, P10D, P24_10, lluvia, isoyeta, etc.);
    2) numbers followed by 'mm';
    3) the last plausible number in the feature name/description when an isohyet
       keyword is present;
    4) a name that is exactly numeric.
    """
    # 0) ArcGIS/Google Earth HTML table descriptions, e.g. VALOR_MM 90.
    html_val = _html_table_value(description)
    if html_val is not None:
        return float(html_val)

    # 1) Explicit attribute keys.
    key_patterns = r"^(p_mm|p10d|p24_?10|p24|precip|precipitacion|precipitación|lluvia|isoyeta|rain|valor|value|valor_mm|mm)$"
    for k, v in (props or {}).items():
        if re.search(key_patterns, str(k).strip(), flags=re.I):
            nums = [x for x in _numbers_from_text(str(v)) if _plausible_precip(x)]
            if nums:
                return float(nums[-1])
    # 2) Any attribute whose key contains a precipitation keyword.
    for k, v in (props or {}).items():
        if re.search(r"p24|p10|precip|lluvia|isoyeta|rain|mm", str(k), flags=re.I):
            nums = [x for x in _numbers_from_text(str(v)) if _plausible_precip(x)]
            if nums:
                return float(nums[-1])

    candidate_texts = [name or "", description or ""] + [str(v) for v in (props or {}).values()]

    # 3) Prefer values explicitly followed by mm.
    for txt in candidate_texts:
        for m in re.finditer(r"(-?\d+(?:[\.,]\d+)?)\s*(?:mm|milimetros|milímetros)\b", txt, flags=re.I):
            val = float(m.group(1).replace(",", "."))
            if _plausible_precip(val):
                return val

    # 4) Text with isohyet/precip keyword: select last plausible number, not the first.
    for txt in candidate_texts:
        if re.search(r"isoyeta|p24|p10d|precip|precipitaci|lluvia|rain", txt or "", flags=re.I):
            nums = [x for x in _numbers_from_text(txt) if _plausible_precip(x)]
            if nums:
                # Avoid choosing return period 2/5/10/25/50/100 when another value follows.
                return float(nums[-1])

    # 5) Fallback: if the feature name is basically just a number, treat it as mm.
    raw_name = (name or "").strip()
    if re.fullmatch(r"-?\d+(?:[\.,]\d+)?", raw_name):
        val = float(raw_name.replace(",", "."))
        if _plausible_precip(val):
            return val
    return None

def isohyet_features_from_kml(kml_text: str) -> List[Feature]:
    """Read a KMZ/KML as isohyet geometries and attach 'p_mm'.

    Supports line isohyets and polygon isohyet bands. Polygon boundaries are used
    for distance-to-control calculations. Features without a detectable value are
    retained with p_mm=None so the user can see that they were read.
    """
    feats = parse_kml_features(kml_text)
    out: List[Feature] = []
    for f in feats:
        if f.ftype not in ("line", "polygon"):
            continue
        p_mm = precipitation_from_text(f.name, f.description, f.properties)
        if p_mm is None:
            z_nonzero = [z for z in f.z_values if abs(z) > 0.001]
            if len(z_nonzero) >= 2 and np.nanstd(z_nonzero) < 0.5:
                z_val = float(np.nanmedian(z_nonzero))
                if _plausible_precip(z_val):
                    p_mm = z_val
        props = dict(f.properties or {})
        props["p_mm"] = p_mm
        # Use polygon boundary for nearest-isohyet distance and mapping as linework.
        geom = f.geometry.boundary if f.ftype == "polygon" else f.geometry
        if geom is None or geom.is_empty:
            continue
        out.append(Feature("line", f.name, f.description, geom, f.z_values, props))
    return out

def nearest_isohyet_to_point(isohyets: List[Feature], point_geom: Optional[BaseGeometry], epsg: int) -> Dict[str, Any]:
    """Return nearest isohyet line to a control/outlet point in projected meters."""
    result = {"ok": False, "p_mm": np.nan, "distance_m": np.nan, "name": "", "n_isohyets": len(isohyets or []), "message": ""}
    if point_geom is None or getattr(point_geom, "is_empty", False):
        result["message"] = "No hay punto de control/descarga definido. Marque el punto en el mapa o active la descarga automática."
        return result
    valid = []
    for f in (isohyets or []):
        p = (f.properties or {}).get("p_mm")
        try:
            p = float(p)
        except Exception:
            p = np.nan
        if _plausible_precip(p) and f.geometry is not None and not f.geometry.is_empty:
            valid.append(f)
    if not valid:
        result["message"] = "Se leyeron geometrías de isoyetas, pero no se detectaron valores de precipitación. Revise nombre, descripción, atributos o coordenada Z."
        return result
    try:
        p_m = project_geom(point_geom, epsg)
    except Exception as e:
        result["message"] = f"No fue posible proyectar el punto de control: {e}"
        return result
    best = None
    best_d = float("inf")
    for f in valid:
        try:
            g = f.geometry
            if not g.is_valid:
                g = g.buffer(0)
            g_m = project_geom(g, epsg)
            d = p_m.distance(g_m)
            if np.isfinite(d) and d < best_d:
                best_d = d
                best = f
        except Exception:
            continue
    if best is None:
        result["message"] = "No fue posible calcular distancia a isoyetas válidas. Puede ingresar P10D manualmente."
        return result
    result.update({
        "ok": True,
        "p_mm": float(best.properties.get("p_mm")),
        "distance_m": float(best_d),
        "name": best.name or "Isoyeta sin nombre",
        "feature": best,
        "message": "Isoyeta más cercana al punto de control detectada correctamente.",
    })
    return result

def parse_kml_features(kml_text: str) -> List[Feature]:
    clean_text = _sanitize_kml_xml(kml_text)
    try:
        root = ET.fromstring(clean_text.encode("utf-8"))
    except ET.ParseError as e1:
        # Some third-party KMZs still fail after declaring missing prefixes.
        # Try a last-resort prefix stripping pass before reporting the error.
        try:
            root = ET.fromstring(_strip_xml_prefixes(clean_text).encode("utf-8"))
        except ET.ParseError as e2:
            raise ValueError(f"El KML no tiene una estructura XML válida: {e1}. Reintento sin prefijos: {e2}")
    placemarks = [el for el in root.iter() if _strip_ns(el.tag) == "Placemark"]
    features: List[Feature] = []

    for pm in placemarks:
        name = _text_or_empty(_find_first(pm, "name"))
        description = _text_or_empty(_find_first(pm, "description"))
        props = _extract_extended_data(pm)

        # Polygons
        for poly_el in _find_all(pm, "Polygon"):
            rings = _find_all(poly_el, "LinearRing")
            if not rings:
                continue
            outer_coords_el = _find_first(rings[0], "coordinates")
            coords, zs = parse_coordinates(_text_or_empty(outer_coords_el))
            if len(coords) >= 4:
                # remove duplicate closes not harmful
                try:
                    geom = Polygon(coords)
                    if geom.is_valid and not geom.is_empty:
                        features.append(Feature("polygon", name, description, geom, zs, props.copy()))
                except Exception:
                    pass

        # Linestrings
        for line_el in _find_all(pm, "LineString"):
            coords_el = _find_first(line_el, "coordinates")
            coords, zs = parse_coordinates(_text_or_empty(coords_el))
            if len(coords) >= 2:
                try:
                    geom = LineString(coords)
                    if geom.is_valid and not geom.is_empty:
                        elev = elevation_from_text(name, description, props)
                        if elev is not None and (not zs or max(abs(z) for z in zs) == 0):
                            zs = [elev] * len(coords)
                        features.append(Feature("line", name, description, geom, zs, props.copy()))
                except Exception:
                    pass

        # gx:Track / gx:MultiTrack geometries, occasionally exported by GIS tools.
        for track_el in _find_all(pm, "Track"):
            coords, zs = parse_gx_coords(track_el)
            if len(coords) >= 2:
                try:
                    geom = LineString(coords)
                    if geom.is_valid and not geom.is_empty:
                        elev = elevation_from_text(name, description, props)
                        if elev is not None and (not zs or max(abs(z) for z in zs) == 0):
                            zs = [elev] * len(coords)
                        features.append(Feature("line", name, description, geom, zs, props.copy()))
                except Exception:
                    pass

        # Points
        for pt_el in _find_all(pm, "Point"):
            coords_el = _find_first(pt_el, "coordinates")
            coords, zs = parse_coordinates(_text_or_empty(coords_el))
            if coords:
                geom = Point(coords[0])
                features.append(Feature("point", name, description, geom, zs, props.copy()))
    return features


def classify_features(features: List[Feature]) -> Dict[str, Any]:
    polygons = [f for f in features if f.ftype == "polygon"]
    lines = [f for f in features if f.ftype == "line"]
    points = [f for f in features if f.ftype == "point"]

    basin = max(polygons, key=lambda f: f.geometry.area) if polygons else None

    contour_lines = []
    channel_lines = []
    other_lines = []
    channel_patterns = r"cauce|eje|hidraul|thalweg|talweg|rio|río|quebrada|main|canal|estero"
    for f in lines:
        z_nonzero = [z for z in f.z_values if abs(z) > 0.001]
        is_contour_by_z = len(z_nonzero) >= 2 and np.nanstd(z_nonzero) < 0.5
        is_channel_by_name = bool(re.search(channel_patterns, f.name or "", re.I))
        if is_channel_by_name and not is_contour_by_z:
            channel_lines.append(f)
        elif is_contour_by_z:
            contour_lines.append(f)
        elif is_channel_by_name:
            channel_lines.append(f)
        else:
            other_lines.append(f)

    outlet_points = []
    outlet_pat = r"salida|descarga|control|outlet|punto.*control|exutorio"
    for f in points:
        if re.search(outlet_pat, f.name or "", re.I):
            outlet_points.append(f)

    return {
        "basin": basin,
        "polygons": polygons,
        "contours": contour_lines,
        "channels": channel_lines,
        "points": points,
        "outlets": outlet_points,
        "other_lines": other_lines,
    }


# -------------------------------
# GIS and metrics
# -------------------------------

def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180) / 6) + 1)
    return (32700 if lat < 0 else 32600) + zone


def project_geom(geom: BaseGeometry, epsg: int) -> BaseGeometry:
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return transform(transformer.transform, geom)


def inverse_project_point(point: Point, epsg: int) -> Point:
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return transform(transformer.transform, point)


def inverse_project_geom(geom: BaseGeometry, epsg: int) -> BaseGeometry:
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return transform(transformer.transform, geom)


def max_distance_between_hull_vertices(poly_m: Polygon) -> float:
    hull = poly_m.convex_hull
    coords = list(hull.exterior.coords)
    if len(coords) <= 1:
        return 0.0
    arr = np.array(coords)
    # O(n^2) acceptable for hull vertices in normal KMZ
    maxd2 = 0.0
    n = len(arr)
    for i in range(n):
        dx = arr[i+1:, 0] - arr[i, 0]
        dy = arr[i+1:, 1] - arr[i, 1]
        if len(dx):
            d2 = np.max(dx*dx + dy*dy)
            if d2 > maxd2:
                maxd2 = d2
    return float(math.sqrt(maxd2))


def compute_basin_metrics(basin_feature: Feature) -> BasinMetrics:
    basin = basin_feature.geometry
    centroid = basin.centroid
    epsg = utm_epsg_from_lonlat(centroid.x, centroid.y)
    basin_m = project_geom(basin, epsg)
    area_m2 = basin_m.area
    perimeter_m = basin_m.length
    max_len_m = max_distance_between_hull_vertices(basin_m)
    minx, miny, maxx, maxy = basin_m.bounds
    bbox_length_m = max(maxx - minx, maxy - miny)
    bbox_width_m = min(maxx - minx, maxy - miny)

    A = area_m2 / 1e6
    P = perimeter_m / 1000
    L = max_len_m / 1000 if max_len_m > 0 else np.nan
    mean_width = A / L if L and L > 0 else np.nan
    kc = P / (2 * math.sqrt(math.pi * A)) if A > 0 else np.nan
    ff = A / (L ** 2) if L and L > 0 else np.nan
    re = (2 * math.sqrt(A / math.pi) / L) if L and L > 0 else np.nan

    return BasinMetrics(
        area_km2=A,
        area_ha=area_m2 / 10000,
        perimeter_km=P,
        centroid_lon=centroid.x,
        centroid_lat=centroid.y,
        epsg_utm=epsg,
        max_geom_length_km=L,
        bbox_length_km=bbox_length_m / 1000,
        bbox_width_km=bbox_width_m / 1000,
        mean_width_km=mean_width,
        compactness_kc=kc,
        form_factor=ff,
        elongation_ratio=re,
    )


def contour_elevation(feature: Feature) -> Optional[float]:
    z_nonzero = [z for z in feature.z_values if abs(z) > 0.001]
    if z_nonzero and np.nanstd(z_nonzero) < 0.5:
        return float(np.nanmedian(z_nonzero))
    elev = elevation_from_text(feature.name, feature.description, feature.properties)
    return elev


def compute_contour_metrics(contours: List[Feature]) -> ContourMetrics:
    elevations = []
    n_vertices = 0
    for c in contours:
        z = contour_elevation(c)
        if z is not None:
            elevations.append(round(float(z), 3))
        n_vertices += len(c.geometry.coords)
    uniq = sorted(set(elevations))
    diffs = np.diff(uniq) if len(uniq) > 1 else []
    eq = float(pd.Series(diffs).mode().iloc[0]) if len(diffs) else None
    return ContourMetrics(
        n_contours=len(contours),
        n_levels=len(uniq),
        z_min=min(uniq) if uniq else None,
        z_max=max(uniq) if uniq else None,
        equidistance_m=eq,
        n_vertices=n_vertices,
    )


def contours_to_points(contours: List[Feature], epsg: int, within: Optional[Polygon] = None) -> Tuple[np.ndarray, np.ndarray]:
    xy = []
    z = []
    for c in contours:
        elev = contour_elevation(c)
        if elev is None:
            continue
        geom_m = project_geom(c.geometry, epsg)
        if within is not None:
            inter = geom_m.intersection(within)
            geoms = []
            if inter.is_empty:
                continue
            if inter.geom_type == "LineString":
                geoms = [inter]
            elif inter.geom_type == "MultiLineString":
                geoms = list(inter.geoms)
            else:
                geoms = [geom_m]
        else:
            geoms = [geom_m]
        for g in geoms:
            if hasattr(g, "coords"):
                for x, y in g.coords:
                    xy.append((x, y)); z.append(elev)
    return np.array(xy, dtype=float), np.array(z, dtype=float)


def idw_interpolate(points_xy: np.ndarray, values_z: np.ndarray, query_xy: np.ndarray, k: int = 8, power: float = 2.0) -> np.ndarray:
    if len(points_xy) == 0:
        return np.full(len(query_xy), np.nan)
    k = min(k, len(points_xy))
    tree = cKDTree(points_xy)
    dist, idx = tree.query(query_xy, k=k)
    if k == 1:
        return values_z[idx]
    dist = np.where(dist < 1e-6, 1e-6, dist)
    w = 1 / (dist ** power)
    vals = values_z[idx]
    return np.sum(w * vals, axis=1) / np.sum(w, axis=1)


def sample_line_profile(channel: LineString, contours: List[Feature], epsg: int, step_m: float = 50.0) -> pd.DataFrame:
    channel_m = project_geom(channel, epsg)
    length = channel_m.length
    if length == 0:
        return pd.DataFrame()
    xy_cont, z_cont = contours_to_points(contours, epsg)
    distances = np.arange(0, length + step_m, step_m)
    if distances[-1] > length:
        distances[-1] = length
    pts = [channel_m.interpolate(float(d)) for d in distances]
    qxy = np.array([(p.x, p.y) for p in pts])
    z = idw_interpolate(xy_cont, z_cont, qxy, k=10, power=2.0) if len(xy_cont) else np.full(len(qxy), np.nan)
    return pd.DataFrame({"dist_m": distances, "cota_m": z})


def estimate_outlet_from_contours(basin_feature: Feature, contours: List[Feature]) -> Optional[Point]:
    """Approximate outlet as point where lowest contour is closest/intersects basin boundary."""
    if not contours:
        return None
    elevations = [(contour_elevation(c), c) for c in contours]
    elevations = [(z, c) for z, c in elevations if z is not None]
    if not elevations:
        return None
    zmin = min(z for z, _ in elevations)
    low_contours = [c for z, c in elevations if abs(z - zmin) < 0.01]
    boundary = basin_feature.geometry.boundary
    candidates = []
    for c in low_contours:
        inter = c.geometry.intersection(boundary)
        if not inter.is_empty:
            if inter.geom_type == "Point": candidates.append(inter)
            elif inter.geom_type == "MultiPoint": candidates.extend(list(inter.geoms))
            elif inter.geom_type in ["LineString", "MultiLineString"]:
                # fallback to representative point
                candidates.append(inter.representative_point())
        else:
            # closest point by scanning vertices
            for lon, lat in c.geometry.coords:
                p = Point(lon, lat)
                candidates.append(p)
    if not candidates:
        return None
    # choose candidate nearest to basin boundary if not exact
    best = min(candidates, key=lambda p: p.distance(boundary))
    return best




def estimate_channel_from_contours(
    basin_feature: Feature,
    contours: List[Feature],
    outlet: Optional[Point] = None,
    high_percentile: float = 92.0,
) -> Tuple[Optional[LineString], Dict[str, Any]]:
    """Estimate a preliminary main channel when no user-defined channel exists.

    This is NOT a hydrologically definitive drainage extraction. It selects a high-elevation
    contour point far from the outlet and joins it to the outlet as a preliminary hydraulic axis.
    The application labels it as an automatic trace that must be validated or replaced by the user.
    """
    info = {"source": "auto_contours", "confidence": "baja", "message": "Sin eje ingresado; eje preliminar estimado desde cotas y punto de salida."}
    if not contours:
        info["message"] = "No hay curvas de nivel para estimar eje automático."
        return None, info
    metrics = compute_basin_metrics(basin_feature)
    epsg = metrics.epsg_utm
    basin_m = project_geom(basin_feature.geometry, epsg)
    if outlet is None:
        outlet = estimate_outlet_from_contours(basin_feature, contours)
    if outlet is None:
        info["message"] = "No hay punto de descarga para estimar eje automático."
        return None, info
    outlet_m = project_geom(outlet, epsg)
    xy, z = contours_to_points(contours, epsg, within=basin_m)
    if len(xy) < 2 or len(z) < 2:
        info["message"] = "Curvas insuficientes para estimar eje automático."
        return None, info
    zthr = np.nanpercentile(z, high_percentile)
    candidates = xy[z >= zthr]
    cand_z = z[z >= zthr]
    if len(candidates) == 0:
        candidates = xy
        cand_z = z
    dx = candidates[:, 0] - outlet_m.x
    dy = candidates[:, 1] - outlet_m.y
    # favor high points and far points
    dist = np.sqrt(dx * dx + dy * dy)
    score = dist + 0.25 * (cand_z - np.nanmin(z))
    idx = int(np.nanargmax(score))
    head_m = Point(float(candidates[idx, 0]), float(candidates[idx, 1]))
    line_m = LineString([(head_m.x, head_m.y), (outlet_m.x, outlet_m.y)])
    # If segment leaves basin, keep it but warn. Direct line is still useful as a starting estimate.
    if not basin_m.buffer(1.0).contains(line_m):
        info["message"] += " El eje recto puede salir parcialmente del polígono; se recomienda dibujarlo manualmente."
    line_ll = inverse_project_geom(line_m, epsg)
    info.update({
        "head_z_m": float(cand_z[idx]),
        "outlet_lon": float(outlet.x),
        "outlet_lat": float(outlet.y),
        "length_km_prelim": float(line_m.length / 1000.0),
    })
    return line_ll, info


def guess_chile_region_basin(lon: float, lat: float) -> Dict[str, str]:
    """Very light geographic hint. It is a preliminary suggestion, not an official DGA basin overlay."""
    out = {"region": "No determinada", "cuenca_sugerida": "Manual", "zona_dga_sugerida": "Manual", "confianza": "baja"}
    if -72.8 <= lon <= -68.0 and -32.6 <= lat <= -29.0:
        out["region"] = "Coquimbo"
        # Approximate by latitude only; user must confirm with official basin data.
        if lat > -31.25:
            out["cuenca_sugerida"] = "IV-Elqui DGA 1995 (editable)"
            out["zona_dga_sugerida"] = "Ip / Elqui preliminar"
        elif lat > -31.85:
            out["cuenca_sugerida"] = "IV-Limarí DGA 1995 (editable)"
            out["zona_dga_sugerida"] = "Jp / Limarí preliminar"
        else:
            out["cuenca_sugerida"] = "IV-Choapa DGA 1995 (editable)"
            out["zona_dga_sugerida"] = "Kp / Choapa preliminar"
        out["confianza"] = "media-baja; confirmar con cuenca oficial DGA"
    return out


def interpolate_factor(factors: Dict[Any, Any], T: float, method: str = "log") -> float:
    """Return factor at T. Exact if available; otherwise log-linear interpolation."""
    clean = {}
    for k, v in (factors or {}).items():
        try:
            kk = float(k); vv = float(v)
            if np.isfinite(kk) and np.isfinite(vv):
                clean[kk] = vv
        except Exception:
            pass
    if not clean:
        return np.nan
    T = float(T)
    if T in clean:
        return clean[T]
    xs = np.array(sorted(clean.keys()), dtype=float)
    ys = np.array([clean[x] for x in xs], dtype=float)
    if T < xs.min() or T > xs.max():
        # Do not extrapolate aggressively; use nearest available factor.
        return float(ys[np.argmin(np.abs(xs - T))])
    if method == "log":
        return float(np.interp(np.log(T), np.log(xs), ys))
    return float(np.interp(T, xs, ys))


VERNI_KING_PRESETS = {
    "Manual": {"c10": np.nan, "ratios": {}, "const": 0.00618, "m": 1.24, "n": 0.88, "nota": "Ingrese coeficientes regionales oficiales/adoptados."},
    # CT=10 desde Tabla 3.25, Manual DGA 1995. Ratios editables: precargados como referencia operacional.
    "IV-Elqui DGA 1995 (editable)": {"c10": 0.057, "ratios": {2: 0.87, 5: 0.93, 10: 1.00, 25: 1.10, 50: 1.15, 100: 1.20, 200: 1.25}, "const": 0.00618, "m": 1.24, "n": 0.88, "nota": "Confirmar que la cuenca pertenece a Elqui. Factor 200 años referencial/extrapolado; editar si cuenta con tabla oficial."},
    "IV-Limarí DGA 1995 (editable)": {"c10": 0.180, "ratios": {2: 0.56, 5: 0.80, 10: 1.00, 25: 1.27, 50: 1.46, 100: 1.63, 200: 1.80}, "const": 0.00618, "m": 1.24, "n": 0.88, "nota": "Confirmar zona Limarí y P24. Factor 200 años referencial/extrapolado; editable."},
    "IV-Choapa DGA 1995 (editable)": {"c10": 0.200, "ratios": {2: 0.39, 5: 0.72, 10: 1.00, 25: 1.50, 50: 1.87, 100: 2.30, 200: 2.73}, "const": 0.00618, "m": 1.24, "n": 0.88, "nota": "Confirmar zona Choapa y P24. Factor 200 años referencial/extrapolado; editable."},
}

DGA_AC_PRESETS = {
    "Manual": {"mode": "generic", "coef": np.nan, "area_exp": np.nan, "p_exp": np.nan, "factors": {}, "alpha_inst": np.nan, "nota": "Ingrese Qref/alpha/Kinst o fórmula regional."},
    "Zona Ip pluvial III-IV / Elqui-Huasco-Copiapo (editable)": {"mode": "iii_iv_formula", "coef": 1.94e-7, "area_exp": 0.776, "p_exp": 3.108, "factors": {2: 0.43, 5: 0.74, 10: 1.00, 25: 1.36, 50: 1.66, 100: 2.00, 200: 2.34}, "alpha_inst": 1.25, "nota": "Usar solo si zona homogénea corresponde. Factor 200 años referencial/extrapolado; editar si cuenta con tabla oficial."},
    "Zona Jp pluvial Limarí - envolvente máxima informe Punitaqui (editable)": {"mode": "iii_iv_formula", "coef": 1.94e-7, "area_exp": 0.776, "p_exp": 3.108, "factors": {2: 0.30, 5: 0.66, 10: 1.00, 20: 1.61, 25: 1.85, 50: 2.76, 75: 3.42, 100: 3.94, 200: 4.85}, "alpha_inst": 2.14, "nota": "Preset corregido para reproducir el informe Estero Punitaqui: usa serie Máx. Q(T)/Q10 y alfa instantáneo 2,14. T=200 es extrapolado y debe justificarse."},
    "Zona Jp pluvial Limarí - media regional (editable)": {"mode": "iii_iv_formula", "coef": 1.94e-7, "area_exp": 0.776, "p_exp": 3.108, "factors": {2: 0.24, 5: 0.61, 10: 1.00, 20: 1.51, 25: 1.71, 50: 2.41, 75: 2.91, 100: 3.30, 200: 4.19}, "alpha_inst": 2.14, "nota": "Serie media regional Jp; alfa corregido a 2,14 según informe Punitaqui. T=200 referencial/extrapolado; confirmar con tabla oficial."},
    "Zona Jp pluvial Limarí (editable)": {"mode": "iii_iv_formula", "coef": 1.94e-7, "area_exp": 0.776, "p_exp": 3.108, "factors": {2: 0.30, 5: 0.66, 10: 1.00, 20: 1.61, 25: 1.85, 50: 2.76, 75: 3.42, 100: 3.94, 200: 4.85}, "alpha_inst": 2.14, "nota": "Alias conservado por compatibilidad. Preset corregido: serie Máx. Q(T)/Q10 y alfa 2,14 para reproducir Punitaqui; editar si la zona/frecuencia oficial difiere."},
    "Zona Kp pluvial Choapa (editable)": {"mode": "iii_iv_formula", "coef": 1.94e-7, "area_exp": 0.776, "p_exp": 3.108, "factors": {2: 0.24, 5: 0.60, 10: 1.00, 25: 1.82, 50: 2.61, 100: 3.67, 200: 4.73}, "alpha_inst": 1.25, "nota": "Preset de trabajo, confirmar en tabla DGA oficial. Factor 200 años referencial/extrapolado; editable."},
}


def input_diagnostics(
    metrics: BasinMetrics,
    contours: List[Feature],
    channel_geom: Optional[LineString],
    outlet_geom: Optional[Point],
    intensity_by_T: Dict[float, float],
    p24_by_T: Dict[float, float],
    dga_params: Dict[str, Any],
    vk_params: Dict[str, Any],
) -> pd.DataFrame:
    rows = []
    def add(item, estado, detalle, accion=""):
        rows.append({"input": item, "estado": estado, "detalle": detalle, "accion_recomendada": accion})
    add("Polígono de cuenca", "OK", f"Área {metrics.area_km2:.3f} km²", "")
    add("Curvas de nivel", "OK" if contours else "Faltante", f"{len(contours)} curvas detectadas", "Cargar KMZ con curvas si no existen")
    add("Cauce principal", "OK" if channel_geom is not None else "Automático/faltante", "Eje definido o estimado" if channel_geom is not None else "No hay línea validada", "Dibujar o cargar cauce principal")
    add("Punto de descarga", "OK" if outlet_geom is not None else "Faltante", "Punto definido o sugerido" if outlet_geom is not None else "No detectado", "Marcar punto de control")
    add("IDF", "OK" if intensity_by_T else "Faltante", f"{len(intensity_by_T)} periodos con intensidad", "Ingresar intensidades I(T,Tc) si se usará racional")
    add("P24", "OK" if p24_by_T else "Faltante", f"{len(p24_by_T)} periodos con P24", "Ingresar P24,T para HUS/SCS/Verni-King")
    vk_ok = False
    if vk_params.get("mode") == "verni_king_mod":
        vk_ok = pd.notna(vk_params.get("c10", np.nan)) and bool(vk_params.get("ratios"))
    else:
        vk_ok = all(pd.notna(vk_params.get(k, np.nan)) for k in ["k", "m", "n"])
    add("Verni-King", "OK" if vk_ok else "Parametrizar", "Preset/factores disponibles" if vk_ok else "Faltan coeficientes", "Confirmar cuenca oficial y coeficientes")
    dga_ok = bool(dga_params.get("mode") == "iii_iv_formula" and pd.notna(dga_params.get("alpha_inst", np.nan)) and bool(dga_params.get("factors"))) or all(pd.notna(dga_params.get(k, np.nan)) for k in ["qref", "alpha", "kinst"])
    add("DGA-AC", "OK" if dga_ok else "Parametrizar", "Fórmula/factores disponibles" if dga_ok else "Faltan alfa/factores/zona", "Confirmar zona homogénea y alfa instantáneo")
    return pd.DataFrame(rows)

# -------------------------------
# Method detection and hydrology
# -------------------------------

OBJECTIVES = [
    "Quebrada natural / descarga aluvial",
    "Atravieso / alcantarilla / baden",
    "Canal o conduccion",
    "Urbanizacion / aguas lluvias",
    "Embalse pequeno / regulacion",
    "Bocatoma",
    "Estudio comparativo preliminar",
]

STANDARD_LAND_USES = {
    "Matorral natural / suelo semiárido": {"C": 0.45, "CN": 74},
    "Ladera con roca y suelo delgado": {"C": 0.65, "CN": 82},
    "Zona agrícola / terrazas": {"C": 0.35, "CN": 70},
    "Urbano baja densidad": {"C": 0.65, "CN": 85},
    "Urbano impermeable": {"C": 0.85, "CN": 92},
    "Personalizado": {"C": 0.45, "CN": 75},
}


def time_of_concentration_kirpich(L_m: float, S: float) -> float:
    if L_m <= 0 or S <= 0:
        return np.nan
    return 0.0195 * (L_m ** 0.77) * (S ** -0.385)  # min


def time_of_concentration_giandotti(A_km2: float, L_km: float, H_m: float) -> float:
    if A_km2 <= 0 or L_km <= 0 or H_m <= 0:
        return np.nan
    return (4 * math.sqrt(A_km2) + 1.5 * L_km) / (0.8 * math.sqrt(H_m)) * 60  # h -> min approx


def detect_methods(A_km2: float, has_idf: bool, has_p24: bool, has_contours: bool, has_channel: bool, has_fluvio: bool, region: str = "Coquimbo", T_max: float = 100) -> pd.DataFrame:
    rows = []
    def add(m, estado, prioridad, motivo):
        rows.append({"metodo": m, "estado": estado, "prioridad": prioridad, "motivo": motivo})

    if A_km2 <= 20 and has_idf:
        add("Racional", "Verde", "Principal", "Area <= 20 km² e intensidad IDF disponible.")
    elif A_km2 <= 25:
        add("Racional", "Amarillo", "Contraste", "Area cercana al umbral de aplicabilidad; usar con cautela.")
    else:
        add("Racional", "Rojo", "No principal", "Area superior al rango recomendable para racional simple.")

    if A_km2 <= 30 and has_idf:
        add("Racional modificado", "Amarillo", "Complementario", "Permite hidrograma simplificado y volumen aproximado.")
    else:
        add("Racional modificado", "Gris", "Condicionado", "Requiere intensidad o lluvia de diseño temporalizada.")

    if has_contours and has_channel and (has_p24 or has_idf):
        add("HUS sintético", "Verde", "Principal", "Hay morfometría, cauce y lluvia suficiente para hidrograma.")
    elif A_km2 > 20:
        add("HUS sintético", "Amarillo", "Principal condicionado", "Recomendable para cuenca mediana; faltan cauce o lluvia.")
    else:
        add("HUS sintético", "Gris", "Complementario", "Aplicable si se requiere hidrograma.")

    if 20 <= A_km2 <= 10000 and T_max <= 100:
        add("DGA-AC parametrizado", "Verde", "Principal DGA", "Rango general compatible; requiere coeficientes/factores oficiales.")
    elif 20 <= A_km2 <= 10000 and T_max > 100:
        add("DGA-AC parametrizado", "Amarillo", "Principal con advertencia", "Rango de área compatible; el periodo 200 años debe revisarse con factores oficiales o extrapolación justificada.")
    elif A_km2 < 20:
        add("DGA-AC parametrizado", "Amarillo", "Contraste", "Area menor al rango típico; usar solo como contraste si corresponde.")
    else:
        add("DGA-AC parametrizado", "Gris", "Condicionado", "Revisar area, region y periodo de retorno.")

    if 20 <= A_km2 <= 10000 and has_p24:
        add("Verni-King modificado parametrizado", "Verde", "Principal DGA", "Area compatible y P24 disponible; requiere coeficientes oficiales.")
    elif 20 <= A_km2 <= 10000:
        add("Verni-King modificado parametrizado", "Amarillo", "Condicionado", "Aplicable si se ingresa P24 y coeficientes regionales.")
    else:
        add("Verni-King modificado parametrizado", "Gris", "Condicionado", "Revisar rango o informacion pluviometrica.")

    if has_p24:
        add("SCS-CN", "Amarillo", "Complementario", "Util para volumen e hidrograma con CN y tormenta de diseño.")
    else:
        add("SCS-CN", "Gris", "Condicionado", "Requiere precipitacion de evento P24 o hietograma.")

    if has_fluvio:
        add("Analisis fluviometrico / transposicion", "Verde", "Principal si hay serie", "Existe estacion/serie representativa a validar.")
    else:
        add("Analisis fluviometrico / transposicion", "Gris", "Referencia", "Requiere estacion DGA representativa o cuenca vecina comparable.")

    return pd.DataFrame(rows)


def rational_q(C: float, I_mm_h: float, A_km2: float) -> float:
    return 0.278 * C * I_mm_h * A_km2


def scs_effective_precip(P_mm: float, CN: float) -> float:
    if CN <= 0 or CN >= 100:
        return np.nan
    S = 25400.0 / CN - 254.0
    Ia = 0.2 * S
    if P_mm <= Ia:
        return 0.0
    return (P_mm - Ia) ** 2 / (P_mm + 0.8 * S)


def scs_peak_q(A_km2: float, Pe_mm: float, Tc_h: float, storm_duration_h: float) -> float:
    # SCS unit hydrograph: Qp = 0.208 A Pe / Tp, with A km2, Pe mm, Tp hours, Q m3/s
    Tp = max(storm_duration_h / 2.0 + 0.6 * Tc_h, 0.01)
    return 0.208 * A_km2 * Pe_mm / Tp


def hydrograph_triangular(Qp: float, tp_h: float, tb_h: Optional[float] = None, n: int = 80) -> pd.DataFrame:
    if tb_h is None:
        tb_h = max(2.67 * tp_h, tp_h * 2.0)
    t = np.linspace(0, tb_h, n)
    q = np.where(t <= tp_h, Qp * t / max(tp_h, 1e-6), Qp * (1 - (t - tp_h) / max(tb_h - tp_h, 1e-6)))
    q = np.maximum(q, 0)
    return pd.DataFrame({"tiempo_h": t, "caudal_m3s": q})


def compute_hydrology(metrics: BasinMetrics, contour_metrics: ContourMetrics, profile_df: pd.DataFrame, inputs: HydrologyInputs) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    A = metrics.area_km2
    # Determine L and H from profile if available, else from geometry and contours
    if profile_df is not None and len(profile_df) >= 2 and profile_df["cota_m"].notna().any():
        L_m = float(profile_df["dist_m"].max())
        z0 = float(profile_df["cota_m"].iloc[0])
        z1 = float(profile_df["cota_m"].iloc[-1])
        H_m = abs(z0 - z1)
    else:
        L_m = metrics.max_geom_length_km * 1000
        H_m = (contour_metrics.z_max - contour_metrics.z_min) if contour_metrics.z_min is not None and contour_metrics.z_max is not None else np.nan
    S = H_m / L_m if L_m and L_m > 0 and H_m and H_m > 0 else np.nan
    Tc_min = time_of_concentration_kirpich(L_m, S) if not np.isnan(S) else np.nan
    Tc_h = Tc_min / 60.0 if not np.isnan(Tc_min) else 1.0

    rows = []
    hydros = {}
    # DGA-AC formula may require P24_10 even when calculating other T
    P24_10 = inputs.p24_by_T.get(10.0, inputs.p24_by_T.get(10, np.nan))

    for T in inputs.return_periods:
        T = float(T)
        I = inputs.intensity_by_T.get(T, np.nan)
        P24 = inputs.p24_by_T.get(T, np.nan)
        # Racional
        if not np.isnan(I):
            Qr = rational_q(inputs.C, I, A)
            rows.append({"T_anios": T, "metodo": "Racional", "Q_m3s": Qr, "estado": "Calculado", "observacion": "Q=0,278*C*I*A"})
            hydros[f"Racional modificado T{T:g}"] = hydrograph_triangular(Qr, Tc_h, max(3*Tc_h, inputs.storm_duration_h if inputs.storm_duration_h>0 else 3*Tc_h))
            rows.append({"T_anios": T, "metodo": "Racional modificado", "Q_m3s": Qr, "estado": "Calculado", "observacion": "Hidrograma triangular simplificado con Q racional"})
        else:
            rows.append({"T_anios": T, "metodo": "Racional", "Q_m3s": np.nan, "estado": "Pendiente", "observacion": "Falta intensidad IDF I(T,Tc)"})
            rows.append({"T_anios": T, "metodo": "Racional modificado", "Q_m3s": np.nan, "estado": "Pendiente", "observacion": "Falta intensidad IDF o lluvia temporalizada"})

        # SCS-CN and HUS simplificado
        if not np.isnan(P24):
            Pe = scs_effective_precip(P24, inputs.CN)
            Qscs = scs_peak_q(A, Pe, Tc_h, inputs.storm_duration_h)
            rows.append({"T_anios": T, "metodo": "SCS-CN", "Q_m3s": Qscs, "estado": "Calculado", "observacion": f"Pe={Pe:.2f} mm; Qp=0,208*A*Pe/Tp"})
            hydros[f"SCS-CN T{T:g}"] = hydrograph_triangular(Qscs, max(inputs.storm_duration_h/2+0.6*Tc_h, Tc_h), None)
            Qhus = Qscs
            rows.append({"T_anios": T, "metodo": "HUS sintético", "Q_m3s": Qhus, "estado": "Calculado", "observacion": "HUS triangular sintético parametrizado desde Pe y Tc"})
            hydros[f"HUS T{T:g}"] = hydrograph_triangular(Qhus, max(inputs.storm_duration_h/2+0.6*Tc_h, Tc_h), None)
        else:
            rows.append({"T_anios": T, "metodo": "SCS-CN", "Q_m3s": np.nan, "estado": "Pendiente", "observacion": "Falta P24,T o hietograma"})
            rows.append({"T_anios": T, "metodo": "HUS sintético", "Q_m3s": np.nan, "estado": "Pendiente", "observacion": "Falta P24,T o hietograma"})

        # Verni-King. Supports DGA-style preset or generic user formula.
        vk = inputs.verni_king_params or {}
        if vk.get("mode") == "verni_king_mod":
            c10 = vk.get("c10", np.nan)
            ratios = vk.get("ratios", {}) or {}
            const = vk.get("const", 0.00618)
            m = vk.get("m", 1.24)
            n = vk.get("n", 0.88)
            ratio = interpolate_factor(ratios, T)
            if not np.isnan(P24) and pd.notna(c10) and pd.notna(ratio):
                cT = float(c10) * float(ratio)
                Qvk = float(cT) * float(const) * (float(P24) ** float(m)) * (A ** float(n))
                obs_vk = f"Q=C(T)*0,00618*P24^1,24*A^0,88; C(T)={cT:.4f}"
                if T > 100:
                    obs_vk += "; T>100: revisar factor regional, valor referencial/editable"
                rows.append({"T_anios": T, "metodo": "Verni-King modificado", "Q_m3s": Qvk, "estado": "Calculado", "observacion": obs_vk})
            else:
                rows.append({"T_anios": T, "metodo": "Verni-King modificado", "Q_m3s": np.nan, "estado": "Pendiente", "observacion": "Faltan P24,T, C10 o factor C(T)/C10"})
        else:
            k = vk.get("k", np.nan); m = vk.get("m", np.nan); n = vk.get("n", np.nan)
            if not np.isnan(P24) and all([pd.notna(k), pd.notna(m), pd.notna(n)]):
                Qvk = float(k) * (float(P24) ** float(m)) * (A ** float(n))
                rows.append({"T_anios": T, "metodo": "Verni-King modificado parametrizado", "Q_m3s": Qvk, "estado": "Calculado", "observacion": "Q=k*P24^m*A^n; coeficientes ingresados por usuario"})
            else:
                rows.append({"T_anios": T, "metodo": "Verni-King modificado parametrizado", "Q_m3s": np.nan, "estado": "Pendiente", "observacion": "Faltan P24,T o coeficientes k,m,n oficiales/regionales"})

        # DGA-AC. Supports III-IV formula or generic qref formula.
        dga = inputs.dga_ac_params or {}
        if dga.get("mode") == "iii_iv_formula":
            coef = dga.get("coef", 1.94e-7)
            area_exp = dga.get("area_exp", 0.776)
            p_exp = dga.get("p_exp", 3.108)
            alpha_inst = dga.get("alpha_inst", np.nan)
            factors = dga.get("factors", {}) or {}
            ft = interpolate_factor(factors, T)
            if pd.notna(P24_10) and pd.notna(alpha_inst) and pd.notna(ft):
                q10_daily = float(coef) * (A ** float(area_exp)) * (float(P24_10) ** float(p_exp))
                Qdga = q10_daily * float(ft) * float(alpha_inst)
                obs_dga = f"Q10={q10_daily:.3f}; Q=Q10*F_T*alpha_inst; F_T={ft:.3f}"
                if T > 100:
                    obs_dga += "; T>100: revisar factor regional, valor referencial/editable"
                rows.append({"T_anios": T, "metodo": "DGA-AC", "Q_m3s": Qdga, "estado": "Calculado", "observacion": obs_dga})
            else:
                rows.append({"T_anios": T, "metodo": "DGA-AC", "Q_m3s": np.nan, "estado": "Pendiente", "observacion": "Faltan P24,10, alfa instantáneo o factores F_T"})
        else:
            qref = dga.get("qref", np.nan); alpha = dga.get("alpha", np.nan); kinst = dga.get("kinst", np.nan)
            factors = dga.get("factors", {}) or {}
            ft = interpolate_factor(factors, T)
            if all([pd.notna(qref), pd.notna(alpha), pd.notna(kinst), pd.notna(ft)]):
                Qdga = float(qref) * (A ** float(alpha)) * float(ft) * float(kinst)
                rows.append({"T_anios": T, "metodo": "DGA-AC parametrizado", "Q_m3s": Qdga, "estado": "Calculado", "observacion": "Q=qref*A^alpha*F_T*Kinst; parámetros ingresados por usuario"})
            else:
                rows.append({"T_anios": T, "metodo": "DGA-AC parametrizado", "Q_m3s": np.nan, "estado": "Pendiente", "observacion": "Faltan qref, alpha, Kinst o factor de frecuencia F_T"})

    return pd.DataFrame(rows), hydros

def profile_summary(metrics: BasinMetrics, contour_metrics: ContourMetrics, profile_df: pd.DataFrame) -> Dict[str, Any]:
    if profile_df is not None and len(profile_df) >= 2 and profile_df["cota_m"].notna().any():
        L_m = float(profile_df["dist_m"].max())
        z_start = float(profile_df["cota_m"].iloc[0])
        z_end = float(profile_df["cota_m"].iloc[-1])
        H_m = abs(z_start - z_end)
    else:
        L_m = metrics.max_geom_length_km * 1000
        z_start = contour_metrics.z_max or np.nan
        z_end = contour_metrics.z_min or np.nan
        H_m = abs(z_start - z_end) if pd.notna(z_start) and pd.notna(z_end) else np.nan
    S = H_m / L_m if L_m > 0 and pd.notna(H_m) else np.nan
    return {
        "L_m": L_m,
        "L_km": L_m/1000,
        "z_start_m": z_start,
        "z_end_m": z_end,
        "H_m": H_m,
        "S_mm": S,
        "S_pct": S*100 if pd.notna(S) else np.nan,
        "Tc_kirpich_min": time_of_concentration_kirpich(L_m, S) if pd.notna(S) and S>0 else np.nan,
        "Tc_giandotti_min": time_of_concentration_giandotti(metrics.area_km2, L_m/1000, H_m) if pd.notna(H_m) and H_m>0 else np.nan,
    }


def to_dict_dataclass(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return dict(obj)

# -------------------------------
# Módulo sedimentológico / caudal detrítico
# -------------------------------

DEBRIS_CV_CLASSES = [
    {"categoria": "Avenida de agua", "cv_min": 0.00, "cv_max": 0.20, "descripcion": "Flujo de agua convencional con transporte de sedimentos suspendidos y de fondo."},
    {"categoria": "Avenida de barro baja", "cv_min": 0.20, "cv_max": 0.30, "descripcion": "Ondas claras; partículas de fondo cercanas a movimiento incipiente."},
    {"categoria": "Avenida de barro media", "cv_min": 0.30, "cv_max": 0.35, "descripcion": "Separación de agua en superficie; arenas y gravas sedimentan y se mueven como arrastre de fondo."},
    {"categoria": "Avenida de barro alta", "cv_min": 0.35, "cv_max": 0.40, "descripcion": "Sedimentación marcada de gravas y guijarros; superficie con dos fases fluidas."},
    {"categoria": "Avenida de barro muy alta", "cv_min": 0.40, "cv_max": 0.45, "descripcion": "Mezcla fácil; propiedades de fluido en deformación; partículas grandes depositan."},
    {"categoria": "Flujo de barro cohesivo", "cv_min": 0.45, "cv_max": 0.48, "descripcion": "Flujo cohesivo que se expande sobre superficies; existe mezcla parcial."},
    {"categoria": "Flujo de barro", "cv_min": 0.48, "cv_max": 0.55, "descripcion": "Deformación plástica bajo su propio peso; cohesivo; no se expande fácilmente."},
    {"categoria": "Deslizamiento / flujo extremadamente denso", "cv_min": 0.55, "cv_max": 0.80, "descripcion": "Falla por deslizamiento en bloque o con deformación interna; ya no se asimila a avenida líquida convencional."},
]


def debris_cv_classes_dataframe() -> pd.DataFrame:
    return pd.DataFrame(DEBRIS_CV_CLASSES)


def classify_debris_flow_cv(cv: float) -> str:
    try:
        c = float(cv)
    except Exception:
        return "Sin clasificar"
    for row in DEBRIS_CV_CLASSES:
        if c >= row["cv_min"] and c < row["cv_max"]:
            return row["categoria"]
    if c >= 0.80:
        return "Concentración fuera de rango hidráulico usual"
    return "Sin clasificar"


def takahashi_cv(slope_pct: float, rho_fluid: float = 1.35, sigma_solid: float = 2.65, phi_deg: float = 30.0, cv_limit: float = 0.80) -> float:
    """Concentración volumétrica crítica según forma simplificada de Takahashi.

    Cv = (rho/(sigma-rho)) * tan(theta) / (tan(phi)-tan(theta))
    donde theta se obtiene desde la pendiente del cauce.
    Se limita a [0, cv_limit] para evitar resultados físicamente no utilizables en la app.
    """
    try:
        S = float(slope_pct) / 100.0
        rho = float(rho_fluid)
        sigma = float(sigma_solid)
        phi = math.radians(float(phi_deg))
        tan_theta = S  # para pendientes de cauce S=tan(theta)
        denom = math.tan(phi) - tan_theta
        if S <= 0 or sigma <= rho or denom <= 0:
            return float("nan")
        cv = (rho / (sigma - rho)) * (tan_theta / denom)
        return float(max(0.0, min(cv, cv_limit)))
    except Exception:
        return float("nan")


def debris_discharge_table(results_df: pd.DataFrame, cv_scenarios: Dict[str, float], only_methods: Optional[List[str]] = None) -> pd.DataFrame:
    """Transforma caudales líquidos QL en caudal sólido QS y caudal detrítico QD."""
    rows = []
    if results_df is None or results_df.empty:
        return pd.DataFrame(rows)
    df = results_df.copy()
    if only_methods:
        df = df[df["metodo"].isin(only_methods)]
    df = df[pd.to_numeric(df.get("Q_m3s"), errors="coerce").notna()]
    for _, r in df.iterrows():
        ql = float(r["Q_m3s"])
        if ql < 0:
            continue
        for name, cv in (cv_scenarios or {}).items():
            try:
                cvf = float(cv)
            except Exception:
                continue
            if cvf <= 0 or cvf >= 0.80:
                qd = float("nan")
                qs = float("nan")
                bf = float("nan")
                estado = "Cv fuera de rango"
            else:
                bf = 1.0 / (1.0 - cvf)
                qd = ql * bf
                qs = qd - ql
                estado = "Calculado"
            rows.append({
                "T_anios": float(r.get("T_anios", float("nan"))),
                "metodo_hidrologico": r.get("metodo", ""),
                "escenario_Cv": name,
                "QL_m3s": ql,
                "Cv": cvf,
                "factor_abultamiento_BF": bf,
                "QS_m3s": qs,
                "QD_m3s": qd,
                "categoria_flujo": classify_debris_flow_cv(cvf),
                "estado": estado,
            })
    return pd.DataFrame(rows)


def sediment_volume_estimates(area_km2: float, channel_length_km: float, prod_cuenca_m3_km2: float = 8000.0, prod_cauce_m3_km: float = 15000.0, manual_volume_m3: float = 0.0) -> pd.DataFrame:
    rows = []
    try:
        A = float(area_km2)
        L = float(channel_length_km)
    except Exception:
        A, L = float("nan"), float("nan")
    if pd.notna(A) and A > 0 and prod_cuenca_m3_km2 and prod_cuenca_m3_km2 > 0:
        rows.append({"metodo": "Productividad por área de cuenca", "volumen_sedimento_m3": A * float(prod_cuenca_m3_km2), "parametro": f"{prod_cuenca_m3_km2:,.0f} m³/km²"})
    if pd.notna(L) and L > 0 and prod_cauce_m3_km and prod_cauce_m3_km > 0:
        rows.append({"metodo": "Productividad por longitud de cauce", "volumen_sedimento_m3": L * float(prod_cauce_m3_km), "parametro": f"{prod_cauce_m3_km:,.0f} m³/km"})
    if manual_volume_m3 and manual_volume_m3 > 0:
        rows.append({"metodo": "Volumen sedimentario manual", "volumen_sedimento_m3": float(manual_volume_m3), "parametro": "ingresado por usuario"})
    return pd.DataFrame(rows)


def empirical_debris_peak_from_volume(volume_df: pd.DataFrame) -> pd.DataFrame:
    """Relaciones empíricas Qp=f(M) para estimación de caudal detrítico máximo.

    M se expresa en m³ y Qp en m³/s.
    """
    formulas = [
        ("Rickenmann 1999", "Qp = 0,1 · M^(5/6)", lambda M: 0.1 * (M ** (5.0/6.0))),
        ("Mizuyama et al. 1992 - flujo granular", "Qp = 0,135 · M^0,780", lambda M: 0.135 * (M ** 0.780)),
        ("Mizuyama et al. 1992 - flujo de barro", "Qp = 0,0188 · M^0,790", lambda M: 0.0188 * (M ** 0.790)),
        ("Jitousono et al. 1996 - Merapi", "Qp = 0,00558 · M^0,831", lambda M: 0.00558 * (M ** 0.831)),
        ("Jitousono et al. 1996 - Sakurajima", "Qp = 0,00135 · M^0,870", lambda M: 0.00135 * (M ** 0.870)),
    ]
    rows = []
    if volume_df is None or volume_df.empty:
        return pd.DataFrame(rows)
    for _, v in volume_df.iterrows():
        try:
            M = float(v["volumen_sedimento_m3"])
        except Exception:
            continue
        if not np.isfinite(M) or M <= 0:
            continue
        for name, formula, fn in formulas:
            try:
                q = float(fn(M))
            except Exception:
                q = float("nan")
            rows.append({
                "volumen_metodo": v.get("metodo", ""),
                "M_m3": M,
                "relacion_empirica": name,
                "formula": formula,
                "Qp_detritico_m3s": q,
                "advertencia": "Relación empírica con alta dispersión; usar solo como contraste o prediseño.",
            })
    return pd.DataFrame(rows)


def debris_susceptibility(metrics: BasinMetrics, contour_metrics: ContourMetrics, prof_summary: Dict[str, Any], material_disponible: str = "Medio", cobertura: str = "Media") -> Dict[str, Any]:
    """Diagnóstico simple de susceptibilidad aluvional/detrítica."""
    score = 0
    reasons = []
    S = prof_summary.get("S_pct", np.nan) if isinstance(prof_summary, dict) else np.nan
    H = prof_summary.get("H_m", np.nan) if isinstance(prof_summary, dict) else np.nan
    A = metrics.area_km2
    if pd.notna(S):
        if S >= 20:
            score += 3; reasons.append("pendiente media de cauce muy alta")
        elif S >= 8:
            score += 2; reasons.append("pendiente media de cauce alta")
        elif S >= 3:
            score += 1; reasons.append("pendiente media de cauce moderada")
    if pd.notna(H):
        if H >= 800:
            score += 2; reasons.append("desnivel hidráulico importante")
        elif H >= 300:
            score += 1; reasons.append("desnivel hidráulico moderado")
    if A <= 5:
        score += 1; reasons.append("cuenca pequeña con respuesta concentrada")
    elif A <= 30:
        score += 1; reasons.append("cuenca de tamaño medio con quebradas potencialmente torrenciales")
    md = (material_disponible or "").lower()
    if "alto" in md:
        score += 3; reasons.append("alta disponibilidad de material detrítico")
    elif "medio" in md:
        score += 1; reasons.append("disponibilidad media de material detrítico")
    cov = (cobertura or "").lower()
    if "baja" in cov or "escasa" in cov:
        score += 2; reasons.append("baja cobertura vegetal")
    elif "media" in cov:
        score += 1; reasons.append("cobertura vegetal media")
    if score >= 7:
        nivel = "Alto"
    elif score >= 4:
        nivel = "Medio"
    else:
        nivel = "Bajo"
    return {"nivel": nivel, "puntaje": score, "fundamentos": "; ".join(reasons) if reasons else "Sin factores críticos automáticos detectados"}
