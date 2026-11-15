from pathlib import Path
import csv
import numpy as np
import rasterio
import geopandas as gpd
from rasterstats import zonal_stats
from shapely.geometry import Polygon
import h3

PROJECT = Path(__file__).resolve().parents[1]

DEM_PATH = PROJECT / "data" / "dem" / "nh44_dem_utm43n.tif"
HEX_PATH = PROJECT / "data" / "corridor_hexes.csv"
OUTPUT_PATH = PROJECT / "data" / "hex_slope.csv"

# 1. Read H3 cells and construct their polygons
rows = list(csv.DictReader(HEX_PATH.open(encoding="utf-8-sig")))

features = []
for row in rows:
    boundary = h3.cell_to_boundary(row["h3_index"])
    polygon = Polygon([(lon, lat) for lat, lon in boundary])

    features.append({
        "h3_index": row["h3_index"],
        "zone_id": row["zone_id"],
        "geometry": polygon
    })

hexes = gpd.GeoDataFrame(features, crs="EPSG:4326")
hexes = hexes.to_crs("EPSG:32643")

# 2. Calculate slope raster in degrees
slope_path = PROJECT / "data" / "dem" / "nh44_slope_degrees.tif"

with rasterio.open(DEM_PATH) as src:
    dem = src.read(1, masked=True).astype("float32")
    xres, yres = src.res

    elevation = dem.filled(np.nan)

    # Calculate elevation gradient in metres/metre
    valid = np.isfinite(elevation)
    profile = src.profile.copy()
# Fill missing elevation pixels temporarily using nearest valid values.
# This prevents NaNs from spreading through the gradient calculation.
from scipy.ndimage import distance_transform_edt, binary_dilation

nearest_indices = distance_transform_edt(
    ~valid,
    return_distances=False,
    return_indices=True
)

filled_elevation = elevation[tuple(nearest_indices)]

dz_dy, dz_dx = np.gradient(filled_elevation, yres, xres)

slope = np.degrees(
    np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
).astype("float32")

# Restore NoData wherever the original DEM was invalid.
invalid_neighbourhood = binary_dilation(
    ~valid,
    structure=np.ones((3, 3), dtype=bool)
)

slope[invalid_neighbourhood] = -9999.0
profile.update(
    dtype="float32",
    count=1,
    nodata=-9999.0,
    compress="deflate"
)

with rasterio.open(slope_path, "w", **profile) as dst:
    dst.write(slope, 1)

print("Slope raster saved:", slope_path)

# 3. Calculate mean slope inside each H3 polygon
stats = zonal_stats(
    hexes,
    str(slope_path),
    stats=["mean", "count"],
    nodata=-9999.0,
    all_touched=False
)

# 4. Export results
with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "h3_index",
        "zone_id",
        "mean_slope_deg",
        "valid_pixel_count"
    ])

    for row, result in zip(rows, stats):
        mean_slope = result["mean"]

        writer.writerow([
            row["h3_index"],
            row["zone_id"],
            round(mean_slope, 4) if mean_slope is not None else "",
            result["count"]
        ])

print("Slope CSV saved:", OUTPUT_PATH)
print("Total H3 cells:", len(rows))
print("Cells with valid slope:", sum(s["mean"] is not None for s in stats))
print("Cells without valid slope:", sum(s["mean"] is None for s in stats))