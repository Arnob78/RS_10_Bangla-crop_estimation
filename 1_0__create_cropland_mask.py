
import os
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling

# --- Configuration ---
SCRIPT_NAME = "1__create_cropland_mask"

# Input file
LULC_PATH = "C:/Users/NagaiLab/GC_Project_work/RS_10_Bangla-crop_estimation/data/raster/LULC/BD_S2_LULC_2021.tif"

# Reference aligned raster for target dimensions
REF_ALIGNED_RASTER_PATH = "C:/Users/NagaiLab/GC_Project_work/RS_10_Bangla-crop_estimation/2__prepare_dl_data_GPU/aligned_climate/bd_monthly_climate_2001.tif"

# --- Script Execution ---

# 1. Create output directory
print(f"Creating output directory: {SCRIPT_NAME}")
OUTPUT_DIR = os.path.join(os.getcwd(), SCRIPT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_MASK_PATH = os.path.join(OUTPUT_DIR, "cropland_mask.tif")

# 2. Read the source LULC data
print(f"Reading LULC data from: {LULC_PATH}")
try:
    with rasterio.open(LULC_PATH) as src_lulc:
        lulc_data = src_lulc.read(1)
        lulc_meta = src_lulc.meta.copy()
except rasterio.errors.RasterioIOError as e:
    print(f"FATAL ERROR: Could not read the source LULC file. Please ensure it is at the correct path. Details: {e}")
    exit()

# 3. Create the binary mask from LULC
print("Creating cropland mask (where pixel value is 40)...")
cropland_mask_original = np.where(lulc_data == 40, 1, 0).astype(rasterio.uint8)

# 4. Get reference metadata from an aligned raster
print(f"Getting reference grid from aligned raster: {REF_ALIGNED_RASTER_PATH}")
try:
    with rasterio.open(REF_ALIGNED_RASTER_PATH) as ref_src:
        ref_meta = ref_src.meta.copy()
        ref_shape = ref_src.shape
except rasterio.errors.RasterioIOError as e:
    print(f"FATAL ERROR: Could not read the reference aligned raster. Please ensure it is at the correct path. Details: {e}")
    exit()

# 5. Reproject the cropland mask to the reference aligned grid
print("Reprojecting cropland mask to match aligned data dimensions...")
aligned_cropland_mask = np.zeros(ref_shape, dtype=rasterio.uint8)

reproject(
    source=cropland_mask_original, # Pass the numpy array directly
    destination=aligned_cropland_mask,
    src_transform=lulc_meta['transform'],
    src_crs=lulc_meta['crs'],
    dst_transform=ref_meta['transform'],
    dst_crs=ref_meta['crs'],
    resampling=Resampling.nearest # Use nearest for categorical data like masks
)

# 6. Update metadata and save the new raster
meta = ref_meta.copy()
meta.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=None)

print(f"Saving aligned mask to: {OUTPUT_MASK_PATH}")
with rasterio.open(OUTPUT_MASK_PATH, 'w', **meta) as dst:
    dst.write(aligned_cropland_mask, 1)

# 7. Verification Step
print("Verifying file creation...")
if os.path.exists(OUTPUT_MASK_PATH):
    print(f"SUCCESS: The file '{os.path.basename(OUTPUT_MASK_PATH)}' was created successfully in the '{SCRIPT_NAME}' directory.")
else:
    print(f"FAILURE: The file '{os.path.basename(OUTPUT_MASK_PATH)}' was NOT created.")

