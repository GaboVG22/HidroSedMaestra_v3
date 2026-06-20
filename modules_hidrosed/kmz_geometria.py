from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import pandas as pd


@dataclass
class KMLGeometrySummary:
    archivo: str
    n_puntos: int
    n_lineas: int
    n_poligonos: int
    n_curvas_con_cota: int
    z_min: float | None = None
    z_max: float | None = None


def read_bytes(source: Any) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if hasattr(source, 'getvalue'):
        return source.getvalue()
    if hasattr(source, 'read'):
        pos = None
        try:
            pos = source.tell()
        except Exception:
            pass
        data = source.read()
        try:
            if pos is not None:
                source.seek(pos)
        except Exception:
            pass
        return data
    raise TypeError('Fuente KML/KMZ no reconocida')


def extract_kml(source: Any, filename: str | None = None) -> str:
    data = read_bytes(source)
    lower = (filename or getattr(source, 'name', '') or str(source)).lower()
    if lower.endswith('.kmz') or zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            kmls = [n for n in zf.namelist() if n.lower().endswith('.kml')]
            if not kmls:
                raise ValueError('El KMZ no contiene KML interno')
            chosen = 'doc.kml' if 'doc.kml' in kmls else kmls[0]
            return zf.read(chosen).decode('utf-8', errors='replace')
    return data.decode('utf-8', errors='replace')


def write_kmz_from_kml(kml_text: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('doc.kml', kml_text.encode('utf-8'))
    return output_path


def _strip_ns(tag: str) -> str:
    return tag.split('}', 1)[-1] if '}' in tag else tag


def _parse_coord_text(text: str) -> list[tuple[float, float, float | None]]:
    coords = []
    for tok in re.split(r'\s+', (text or '').strip()):
        if not tok:
            continue
        parts = [p for p in tok.split(',') if p != '']
        if len(parts) < 2:
            continue
        try:
            lon = float(parts[0]); lat = float(parts[1]); z = float(parts[2]) if len(parts) >= 3 else None
        except Exception:
            continue
        if -180 <= lon <= 180 and -90 <= lat <= 90:
            coords.append((lon, lat, z))
    return coords


def _name_to_elevation(name: str | None) -> float | None:
    if not name:
        return None
    # Busca cotas tipo 750, Cota 750 m, 750.5, z=750, curva_750
    m = re.search(r'(?<!\d)([-+]?\d{2,5}(?:[\.,]\d+)?)(?:\s*m)?(?!\d)', str(name), flags=re.I)
    if not m:
        return None
    try:
        return float(m.group(1).replace(',', '.'))
    except Exception:
        return None


def _placemark_name(pm: ET.Element) -> str:
    for child in pm.iter():
        if _strip_ns(child.tag) == 'name' and child.text:
            return child.text.strip()
    return 'Sin nombre'


def parse_kml_geometries(source: Any, filename: str | None = None) -> dict[str, pd.DataFrame]:
    """Lee puntos, líneas y polígonos desde KML/KMZ. Mantiene altitud KML y cota detectada desde nombre."""
    kml = extract_kml(source, filename)
    try:
        root = ET.fromstring(kml.encode('utf-8'))
    except ET.ParseError as exc:
        raise ValueError(f'KML inválido: {exc}') from exc
    points, lines, polygons = [], [], []
    line_id = 0; poly_id = 0; point_id = 0
    for pm in root.iter():
        if _strip_ns(pm.tag) != 'Placemark':
            continue
        name = _placemark_name(pm)
        z_name = _name_to_elevation(name)
        # Point / LineString / Polygon by parent tags around coordinates
        for elem in pm.iter():
            tag = _strip_ns(elem.tag)
            if tag != 'coordinates' or not elem.text:
                continue
            # infer geometry by ancestors not available in ElementTree. Use surrounding text from parent search.
            coords = _parse_coord_text(elem.text)
            if not coords:
                continue
            if len(coords) == 1:
                lon, lat, z = coords[0]
                points.append({'id_punto': point_id, 'nombre': name, 'lon': lon, 'lat': lat, 'z_kml': z, 'cota_m': z if z is not None else z_name})
                point_id += 1
            else:
                # if first closes last and many vertices, tag as polygon when parent text contains Polygon in pm xml
                pm_txt = ET.tostring(pm, encoding='unicode', method='xml')
                is_poly = '<Polygon' in pm_txt or '<LinearRing' in pm_txt
                if is_poly:
                    poly_id += 1
                    for idx, (lon, lat, z) in enumerate(coords):
                        polygons.append({'id_poligono': poly_id, 'nombre': name, 'orden': idx, 'lon': lon, 'lat': lat, 'z_kml': z, 'cota_m': z if z is not None else z_name})
                else:
                    line_id += 1
                    for idx, (lon, lat, z) in enumerate(coords):
                        lines.append({'id_linea': line_id, 'nombre': name, 'orden': idx, 'lon': lon, 'lat': lat, 'z_kml': z, 'cota_m': z if z is not None else z_name})
    return {'puntos': pd.DataFrame(points), 'lineas': pd.DataFrame(lines), 'poligonos': pd.DataFrame(polygons), 'kml_text': kml}


def summarize_geometries(geoms: dict[str, pd.DataFrame], archivo: str = '') -> KMLGeometrySummary:
    pts = geoms.get('puntos', pd.DataFrame())
    lns = geoms.get('lineas', pd.DataFrame())
    polys = geoms.get('poligonos', pd.DataFrame())
    cota = pd.concat([df[['cota_m']] for df in [pts, lns, polys] if not df.empty and 'cota_m' in df.columns], ignore_index=True) if any(not df.empty for df in [pts, lns, polys]) else pd.DataFrame()
    vals = pd.to_numeric(cota.get('cota_m', pd.Series(dtype=float)), errors='coerce').dropna() if not cota.empty else pd.Series(dtype=float)
    return KMLGeometrySummary(
        archivo=archivo,
        n_puntos=len(pts),
        n_lineas=int(lns['id_linea'].nunique()) if not lns.empty and 'id_linea' in lns.columns else 0,
        n_poligonos=int(polys['id_poligono'].nunique()) if not polys.empty and 'id_poligono' in polys.columns else 0,
        n_curvas_con_cota=int(lns.loc[pd.to_numeric(lns.get('cota_m', pd.Series(dtype=float)), errors='coerce').notna(), 'id_linea'].nunique()) if not lns.empty and 'id_linea' in lns.columns else 0,
        z_min=float(vals.min()) if len(vals) else None,
        z_max=float(vals.max()) if len(vals) else None,
    )


def merge_kml_documents(kml_sources: list[tuple[str, str]], output_path: Path, name: str) -> Path:
    """Combina documentos KML. kml_sources = [(titulo, kml_text)]."""
    placemarks = []
    styles = []
    for titulo, text in kml_sources:
        styles.extend(re.findall(r'<Style\b[\s\S]*?</Style>', text, flags=re.I))
        styles.extend(re.findall(r'<StyleMap\b[\s\S]*?</StyleMap>', text, flags=re.I))
        for pm in re.findall(r'<Placemark\b[\s\S]*?</Placemark>', text, flags=re.I):
            placemarks.append(f'<Folder><name>{titulo}</name>{pm}</Folder>')
    safe = (name or 'HidroSed KMZ').replace('&', '&amp;').replace('<', '').replace('>', '')
    combined = '<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n'
    combined += f'<name>{safe}</name>\n' + '\n'.join(styles + placemarks) + '\n</Document></kml>\n'
    return write_kmz_from_kml(combined, output_path)


def generate_preliminary_axis_from_point(control_point: dict, bbox: dict | None = None, length_km: float = 5.0, azimuth_deg: float = 180.0) -> str:
    """Genera un KML simple de eje preliminar si no se cargó eje. Es una ayuda gráfica, no reemplaza eje validado."""
    lat0 = float(control_point.get('latitud') or control_point.get('latitude'))
    lon0 = float(control_point.get('longitud') or control_point.get('longitude'))
    # Genera línea centrada en el punto con azimuth. Aproximación local suficiente para visualización.
    L = float(length_km) * 1000.0
    az = math.radians(float(azimuth_deg))
    dx = math.sin(az) * L / 2.0
    dy = math.cos(az) * L / 2.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = max(1.0, 111_320.0 * math.cos(math.radians(lat0)))
    lon1 = lon0 - dx / m_per_deg_lon; lat1 = lat0 - dy / m_per_deg_lat
    lon2 = lon0 + dx / m_per_deg_lon; lat2 = lat0 + dy / m_per_deg_lat
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<name>Eje preliminar HidroSed</name>
<Placemark><name>Eje preliminar automático - revisar</name><LineString><tessellate>1</tessellate><coordinates>
{lon1:.8f},{lat1:.8f},0 {lon0:.8f},{lat0:.8f},0 {lon2:.8f},{lat2:.8f},0
</coordinates></LineString></Placemark>
</Document></kml>'''
