
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

from modules_hidrosed.watershed_delineation import delineate_watershed_from_dem

out = Path("outputs/test_v6_6")
out.mkdir(parents=True, exist_ok=True)
dem = out / "synthetic_dem.tif"

# DEM sintético: valle que drena hacia el borde inferior central.
nrows, ncols = 100, 100
r, c = np.indices((nrows, ncols))
data = 1000 + (nrows - r) * 2 + np.abs(c - 50) * 0.8
data = data.astype("float32")
transform = from_origin(-71.0, -30.0, 0.0003, 0.0003)

with rasterio.open(
    dem, "w", driver="GTiff", height=nrows, width=ncols, count=1, dtype="float32",
    crs="EPSG:4326", transform=transform, nodata=-9999
) as dst:
    dst.write(data, 1)

# Punto de salida cerca de la parte baja del valle.
lon, lat = transform * (50, 95)
res = delineate_watershed_from_dem(dem, lon, lat, out, snap_radius_m=500, max_cells=200000, simplify_m=20)
assert res.kmz_path.exists()
assert res.kml_path.exists()
assert res.area_km2 > 0
print("OK v6.6 watershed", res.area_km2, res.n_cells_basin)
