
from pathlib import Path
import tempfile
import zipfile

from src.hidro_kmz_core import read_kmz_or_kml, parse_kml_features, classify_features, compute_basin_metrics
from modules_hidrosed.dem_downloader import bbox_from_point, split_bbox
from modules_master.kmz_workspace import create_combined_workspace_kmz

ROOT = Path(__file__).parent
kml = read_kmz_or_kml(str(ROOT / 'data' / 'Quebrada_Las_Cardas_2_1.kmz'))
features = parse_kml_features(kml)
classes = classify_features(features)
assert classes['basin'] is not None, 'No se detectó cuenca demo'
metrics = compute_basin_metrics(classes['basin'])
assert metrics.area_km2 > 0, 'Área inválida'

bbox = bbox_from_point(metrics.centroid_lat, metrics.centroid_lon, 0.1)
tiles = split_bbox(bbox, 10)
assert len(tiles) == 10, 'split_bbox no generó 10 teselas'

curve_kml = '<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><name>Curva 100 m</name><LineString><coordinates>-71,-30,100 -71.01,-30.01,100</coordinates></LineString></Placemark></Document></kml>'
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    curve_path = td / 'curvas.kml'
    curve_path.write_text(curve_kml, encoding='utf-8')
    out = td / 'workspace.kmz'
    create_combined_workspace_kmz(ROOT / 'data' / 'Quebrada_Las_Cardas_2_1.kmz', curve_path, out, 'test')
    assert out.exists() and out.stat().st_size > 100
    with zipfile.ZipFile(out) as zf:
        assert 'doc.kml' in zf.namelist()

print('Smoke test maestra OK')
print(f'Área demo: {metrics.area_km2:.2f} km2 | teselas: {len(tiles)}')
