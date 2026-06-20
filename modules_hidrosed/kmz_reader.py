"""Lectura de KMZ/KML para detectar un punto de control.

El módulo no usa Streamlit. Devuelve datos simples para que otras pantallas
puedan reutilizar el mismo resultado.
"""

from __future__ import annotations

import io
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable


@dataclass
class ControlPoint:
    name: str
    latitude: float
    longitude: float
    altitude: float | None = None
    source: str | None = None


class KMLParseError(ValueError):
    """Error controlado para KML/KMZ inválidos."""


def _read_bytes(file_or_bytes: BinaryIO | bytes | Path) -> bytes:
    if isinstance(file_or_bytes, bytes):
        return file_or_bytes
    if isinstance(file_or_bytes, Path):
        return file_or_bytes.read_bytes()
    if hasattr(file_or_bytes, "getvalue"):
        return file_or_bytes.getvalue()
    if hasattr(file_or_bytes, "read"):
        pos = None
        try:
            pos = file_or_bytes.tell()
        except Exception:
            pos = None
        data = file_or_bytes.read()
        if pos is not None:
            try:
                file_or_bytes.seek(pos)
            except Exception:
                pass
        return data
    raise TypeError("Entrada no soportada para lectura de KMZ/KML.")


def extract_kml_text(file_or_bytes: BinaryIO | bytes | Path, filename: str | None = None) -> str:
    """Extrae texto KML desde un archivo .kml o .kmz."""
    data = _read_bytes(file_or_bytes)
    lower_name = (filename or "").lower()

    if lower_name.endswith(".kmz") or zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                raise KMLParseError("El KMZ no contiene ningún archivo KML.")
            # Se prioriza doc.kml si existe, luego el primero encontrado.
            chosen = "doc.kml" if "doc.kml" in kml_names else kml_names[0]
            return zf.read(chosen).decode("utf-8", errors="replace")

    return data.decode("utf-8", errors="replace")


def _strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _iter_coordinate_texts_xml(kml_text: str) -> Iterable[tuple[str | None, str]]:
    """Intenta extraer pares nombre/coordenadas mediante XML."""
    try:
        root = ET.fromstring(kml_text.encode("utf-8"))
    except ET.ParseError:
        return []

    placemarks = []
    for elem in root.iter():
        if _strip_namespace(elem.tag) == "Placemark":
            placemarks.append(elem)

    results: list[tuple[str | None, str]] = []
    if placemarks:
        for pm in placemarks:
            name = None
            coords = []
            for child in pm.iter():
                tag = _strip_namespace(child.tag)
                if tag == "name" and child.text and not name:
                    name = child.text.strip()
                elif tag == "coordinates" and child.text:
                    coords.append(child.text.strip())
            for c in coords:
                results.append((name, c))
    else:
        for elem in root.iter():
            if _strip_namespace(elem.tag) == "coordinates" and elem.text:
                results.append((None, elem.text.strip()))
    return results


def _iter_coordinate_texts_regex(kml_text: str) -> Iterable[tuple[str | None, str]]:
    pattern = re.compile(r"<coordinates[^>]*>(.*?)</coordinates>", re.I | re.S)
    for match in pattern.finditer(kml_text):
        yield None, match.group(1).strip()


def _parse_coordinate_token(token: str) -> tuple[float, float, float | None] | None:
    parts = [p for p in token.strip().split(",") if p != ""]
    if len(parts) < 2:
        return None
    try:
        lon = float(parts[0])
        lat = float(parts[1])
        alt = float(parts[2]) if len(parts) >= 3 else None
    except ValueError:
        return None
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        return None
    return lon, lat, alt


def parse_first_control_point(kml_text: str, source: str | None = None) -> ControlPoint:
    """Devuelve el primer punto/coordenada válida encontrado en un KML."""
    candidates = list(_iter_coordinate_texts_xml(kml_text))
    if not candidates:
        candidates = list(_iter_coordinate_texts_regex(kml_text))

    for name, coord_text in candidates:
        tokens = re.split(r"\s+", coord_text.strip())
        for token in tokens:
            parsed = _parse_coordinate_token(token)
            if parsed:
                lon, lat, alt = parsed
                return ControlPoint(
                    name=name or "Punto de control",
                    latitude=lat,
                    longitude=lon,
                    altitude=alt,
                    source=source,
                )

    raise KMLParseError("No se encontró una coordenada válida en el archivo KML/KMZ.")


def read_control_point(file_or_bytes: BinaryIO | bytes | Path, filename: str | None = None) -> ControlPoint:
    kml_text = extract_kml_text(file_or_bytes, filename=filename)
    return parse_first_control_point(kml_text, source=filename)
