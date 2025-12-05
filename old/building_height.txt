import ee
ee.Initialize()

# -------------------------------
# USER REGION
# -------------------------------
min_lon, min_lat = -116, 34
max_lon, max_lat = -115, 35

region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

# Build dynamic description name
desc = f"Building_Height_{min_lon}_{max_lon}_{min_lat}_{max_lat}".replace("-", "w").replace(".", "")

print("Export description:", desc)

# -------------------------------
# LOAD TILE
# -------------------------------
tile = ee.FeatureCollection(
    "projects/sat-io/open-datasets/GLOBAL_BUILDING_ATLAS/w120_n35_w115_n30"
)

# Clip buildings to region
clipped = tile.filterBounds(region).map(
    lambda f: f.intersection(region, maxError=1)
)

count = clipped.size().getInfo()
print("Buildings in region:", count)

# -------------------------------
# RASTERIZE HEIGHT FIELD
# -------------------------------
# Use "height" column and rasterize at 30 m resolution
height_raster = clipped.reduceToImage(
    properties=["height"],
    reducer=ee.Reducer.first()
).rename("building_height")

# -------------------------------
# EXPORT AS GEOtTIFF
# -------------------------------
task = ee.batch.Export.image.toDrive(
    image=height_raster,
    description=desc,
    folder=None,                # optional
    fileNamePrefix=desc,
    region=region,
    scale=30,
    crs="EPSG:4326",
    fileFormat="GeoTIFF",
)

task.start()

print("TIFF export started. Check Tasks tab.")

This is the python script for building height, watch out for this part  "tile = ee.FeatureCollection(
   "projects/sat-io/open-datasets/GLOBAL_BUILDING_ATLAS/w120_n35_w115_n30"
)" the tile name needs to be updated based on the latitude and longitude, the tile is named for lat/lon box every 5 degree

Data resolution is 3x3 meter and therefore we need to interpolated it to fire model resolution. You can check this website further for more details on their paper https://essd.copernicus.org/preprints/essd-2025-327/essd-2025-327.pdf and GEE website https://gee-community-catalog.org/projects/gba/#coverage-and-accuracy

