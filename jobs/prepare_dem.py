from pathlib import Path
import csv
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import Polygon, mapping
import h3

PROJECT = Path(__file__).resolve().parents[1]
DEM_DIR = Path.home() / "Documents" / "NH44_Copernicus_DEM"
CSV_PATH = PROJECT / "data" / "corridor_hexes.csv"
OUTPUT_DIR = PROJECT / "data" / "dem"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Read the six source DEM tiles
dem_files = sorted(DEM_DIR.glob("Copernicus_DSM_COG_10_*.tif"))
if len(dem_files) != 6:
    raise RuntimeError(f"Expected 6 DEM tiles, found {len(dem_files)}")

sources = [rasterio.open(path) for path in dem_files]

try:
    # 2. Merge the tiles
    mosaic, mosaic_transform = merge(sources)
    profile = sources[0].profile.copy()
    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=mosaic_transform,
        count=1,
        compress="deflate"
    )

    mosaic_path = OUTPUT_DIR / "nh44_dem_mosaic.tif"
    with rasterio.open(mosaic_path, "w", **profile) as dst:
        dst.write(mosaic[0], 1)
finally:
    for src in sources:
        src.close()

print("Mosaic saved:", mosaic_path)

# 3. Build polygons from the actual H3 boundaries
rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8-sig")))
features = []

for row in rows:
    cell = row["h3_index"]
    boundary = h3.cell_to_boundary(cell)  # (latitude, longitude)
    polygon = Polygon([(lon, lat) for lat, lon in boundary])
    features.append({
        "h3_index": cell,
        "zone_id": row["zone_id"],
        "geometry": polygon
    })

hexes = gpd.GeoDataFrame(features, crs="EPSG:4326")
print("H3 cells:", len(hexes))
print("H3 extent:", hexes.total_bounds)

# 4. Reproject the mosaic to UTM Zone 43N
with rasterio.open(mosaic_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs,
        "EPSG:32643",
        src.width,
        src.height,
        *src.bounds,
        resolution=30
    )

    nodata_value = -9999.0

    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        crs="EPSG:32643",
        transform=transform,
        width=width,
        height=height,
        dtype="float32",
        nodata=nodata_value,
        compress="deflate"
    )

    projected_path = OUTPUT_DIR / "nh44_dem_utm43n.tif"

    with rasterio.open(projected_path, "w", **profile) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs="EPSG:32643",
            dst_nodata=nodata_value,
            resampling=Resampling.bilinear,
            init_dest_nodata=True
        )

print("Projected DEM saved:", projected_path)
print("DEM preparation complete.")
