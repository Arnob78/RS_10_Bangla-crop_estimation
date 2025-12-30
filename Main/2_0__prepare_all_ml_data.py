import os
import rasterio
from rasterio.enums import Resampling
import rioxarray
import dask
import numpy as np
from dask.diagnostics import ProgressBar

# --- Configuration ---
SCRIPT_NAME = "2__prepare_dl_data_GPU-modified"
PROJECT_ROOT = r"C:\Users\NagaiLab\GC_Project_work\submitted_RS_10_Bangla-crop_estimation"

# --- Input Directories ---
NDVI_DIR = os.path.join(PROJECT_ROOT, "data/raster/NDVI_monthly")
CLIMATE_DIR = os.path.join(PROJECT_ROOT, "data/raster/Climate_Monthly")
STATIC_DIR = os.path.join(PROJECT_ROOT, "data/raster/static_data")
LULC_DIR = os.path.join(PROJECT_ROOT, "data/raster/LULC")

# Reference raster for alignment
REF_RASTER_PATH = os.path.join(LULC_DIR, "BD_S2_LULC_2021_250m.tif")

# --- Script Execution ---

# 1. Create output directory
OUTPUT_DIR = os.path.join(PROJECT_ROOT, SCRIPT_NAME)
print(f"Creating output directory: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Open the reference raster with rioxarray to be used for matching
print(f"Reading reference grid from: {os.path.basename(REF_RASTER_PATH)}")
ref_rds = rioxarray.open_rasterio(REF_RASTER_PATH, chunks=True)

def align_and_save_raster(file_name, input_dir, output_path, ref_rds):
    """
    Aligns a single raster to the reference grid using rioxarray and dask.
    This function is meant to be wrapped with dask.delayed.
    """
    src_path = os.path.join(input_dir, file_name)
    out_file_path = os.path.join(output_path, file_name)

    # Open the source raster with dask chunks for parallel processing
    src_rds = rioxarray.open_rasterio(src_path, chunks=True)

    # Reproject to match the reference raster.
    # This operation is lazy and will be computed by dask later.
    rds_reprojected = src_rds.rio.reproject_match(ref_rds, resampling=Resampling.bilinear)

    # Write the output raster. This triggers the dask computation.
    # The `compute=False` gives us a delayed object.
    return rds_reprojected.rio.to_raster(out_file_path, compute=False, compress='LZW')


def process_and_save_rasters(input_dir, prefix, output_subdir):
    """Finds, aligns, and saves a set of rasters using dask in batches."""
    output_path = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(output_path, exist_ok=True)

    raster_files = [f for f in os.listdir(input_dir) if f.startswith(prefix) and f.endswith('.tif')]
    print(f"\nFound {len(raster_files)} rasters in '{input_dir}'. Aligning to reference grid...")

    batch_size = 1
    num_batches = (len(raster_files) + batch_size - 1) // batch_size # Ceiling division

    print(f"Processing in {num_batches} batches of size {batch_size}...")

    for i in range(0, len(raster_files), batch_size):
        batch_files = raster_files[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{num_batches}...")

        delayed_tasks = []
        for file_name in batch_files:
            task = dask.delayed(align_and_save_raster)(file_name, input_dir, output_path, ref_rds)
            delayed_tasks.append(task)

        # Compute just the tasks for this batch
        with ProgressBar():
            dask.compute(*delayed_tasks)

    print(f"\nAll {output_subdir} rasters have been aligned and saved in '{output_path}'.")

if __name__ == "__main__":
    # --- Optional GPU setup ---
    # This script can use dask-cuda for GPU acceleration.
    # If dask-cuda is not installed, it will fall back to CPU-based dask parallelism.
    # This setup MUST be inside the __main__ block to prevent multiprocessing errors.
    try:
        import dask_cuda
        from dask.distributed import Client
        cluster = dask_cuda.LocalCUDACluster()
        client = Client(cluster)
        print("dask-cuda found, using GPU for acceleration.")
    except ImportError:
        print("dask-cuda not found. Using CPU-based dask parallelism.")
        # Using dask with the threaded scheduler for I/O bound tasks on a single machine.
        # Limiting the number of workers to 4 to reduce memory consumption.
        # You can adjust this number based on your system's memory.
        dask.config.set(scheduler='threads', num_workers=4)

    # 3. Process all data categories
    process_and_save_rasters(NDVI_DIR, "bd_monthly_ndvi_", "aligned_ndvi")
    target_output_path = os.path.join(OUTPUT_DIR, "target_ndvi_2023.tif")
    # --- Calculate NDVI Anomaly (Z-score) for the target year ---
    print("\nCalculating NDVI anomaly for the target year 2023...")

    try:
        # Get the profile from one of the aligned rasters to use as a template
        with rasterio.open(os.path.join(OUTPUT_DIR, "aligned_ndvi", 'bd_monthly_ndvi_2023.tif')) as ref_raster:
            profile = ref_raster.profile
            profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=-9999)

        # List all the historical NDVI files (2001-2022)
        historical_ndvi_paths = [os.path.join(OUTPUT_DIR, "aligned_ndvi", f'bd_monthly_ndvi_{year}.tif') for year in range(2001, 2023)]
        ndvi_2023_path = os.path.join(OUTPUT_DIR, "aligned_ndvi", 'bd_monthly_ndvi_2023.tif')

        # Stack all historical arrays into a 3D numpy array
        historical_arrays = []
        for path in historical_ndvi_paths:
            with rasterio.open(path) as src:
                historical_arrays.append(src.read(1))
        historical_stack = np.stack(historical_arrays, axis=0)

        # Calculate mean and std dev along the time axis, ignoring NaNs
        print("Calculating historical mean and standard deviation...")
        mean_ndvi = np.nanmean(historical_stack, axis=0)
        std_ndvi = np.nanstd(historical_stack, axis=0)

        # Open the 2023 NDVI raster
        with rasterio.open(ndvi_2023_path) as src_2023:
            ndvi_2023_array = src_2023.read(1)

        # Calculate Z-score (anomaly)
        print("Calculating Z-score anomaly...")
        # Initialize an array with the no-data value
        anomaly_array = np.full_like(mean_ndvi, -9999, dtype=float)

        # Create a mask for valid std_dev values (not 0 and not NaN) to avoid division errors
        valid_std_mask = (std_ndvi != 0) & ~np.isnan(std_ndvi)

        # Calculate anomaly only where std_dev is valid
        anomaly_array[valid_std_mask] = (ndvi_2023_array[valid_std_mask] - mean_ndvi[valid_std_mask]) / std_ndvi[valid_std_mask]

        # Ensure any remaining NaNs from the input data are also set to the no-data value
        anomaly_array[np.isnan(anomaly_array)] = -9999

        # Save the new anomaly raster as the target
        with rasterio.open(target_output_path, 'w', **profile) as dst:
            dst.write(anomaly_array.astype(rasterio.float32), 1)

        print(f"SUCCESS: Target NDVI ANOMALY (Z-score) for 2023 saved to {target_output_path}")

    except Exception as e:
        print(f"ERROR calculating NDVI anomaly: {e}")
        print("Please ensure all NDVI files from 2001-2023 are present and valid in the 'aligned_ndvi' directory.")

    process_and_save_rasters(CLIMATE_DIR, "bd_monthly_climate_", "aligned_climate")
    process_and_save_rasters(STATIC_DIR, "bd_", "aligned_static")



    print(f"\nScript {SCRIPT_NAME}.py finished successfully.")
