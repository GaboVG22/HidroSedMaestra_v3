
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any


def read_kml_text(file_input: Any) -> str:
    """Read KML text from a path, uploaded file, KMZ, or raw bytes."""
    name = getattr(file_input, "name", "") or str(file_input)
    if isinstance(file_input, (str, Path)):
        data = Path(file_input).read_bytes()
        name = str(file_input)
    elif hasattr(file_input, "getvalue"):
        data = file_input.getvalue()
    elif hasattr(file_input, "read"):
        pos = None
        try:
            pos = file_input.tell()
        except Exception:
            pass
        data = file_input.read()
        try:
            if pos is not None:
                file_input.seek(pos)
        except Exception:
            pass
    elif isinstance(file_input, bytes):
        data = file_input
    else:
        raise TypeError("Entrada KML/KMZ no reconocida.")

    if str(name).lower().endswith('.kmz') or data[:2] == b'PK':
        with zipfile.ZipFile(io.BytesIO(data), 'r') as zf:
            kml_names = [n for n in zf.namelist() if n.lower().endswith('.kml')]
            if not kml_names:
                raise ValueError('El KMZ no contiene archivo KML interno.')
            kml_name = 'doc.kml' if 'doc.kml' in kml_names else kml_names[0]
            return zf.read(kml_name).decode('utf-8', errors='replace')
    return data.decode('utf-8', errors='replace')


def _extract_blocks(kml_text: str, tag: str) -> list[str]:
    pattern = rf"<{tag}\b[\s\S]*?</{tag}>"
    return re.findall(pattern, kml_text, flags=re.IGNORECASE)


def create_combined_workspace_kmz(
    basin_kml_source: Any,
    contours_kml_source: Any,
    output_path: Path,
    project_name: str = 'Proyecto HidroSed Hidrología',
) -> Path:
    """Create a KMZ combining basin/control KML with generated DEM contours."""
    basin_text = read_kml_text(basin_kml_source)
    contours_text = read_kml_text(contours_kml_source)

    style_blocks: list[str] = []
    for txt in (basin_text, contours_text):
        style_blocks.extend(_extract_blocks(txt, 'Style'))
        style_blocks.extend(_extract_blocks(txt, 'StyleMap'))

    placemarks: list[str] = []
    placemarks.extend(_extract_blocks(basin_text, 'Placemark'))
    placemarks.extend(_extract_blocks(contours_text, 'Placemark'))
    if not placemarks:
        raise ValueError('No se encontraron Placemark para combinar.')

    safe_name = (project_name or 'Proyecto HidroSed Hidrología')
    safe_name = safe_name.replace('&', '&amp;').replace('<', '').replace('>', '')
    body = "\n".join(style_blocks + placemarks)
    combined = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">\n'
        '<Document>\n'
        f'<name>{safe_name} - KMZ combinado DEM-curvas-cuenca</name>\n'
        f'{body}\n'
        '</Document>\n'
        '</kml>\n'
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('doc.kml', combined.encode('utf-8'))
    return output_path
