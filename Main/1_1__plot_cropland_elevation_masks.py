import os
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings('ignore')

def add_manual_scalebar(ax, x_frac=0.05, y_frac=0.12, length_km=100, color='black', fontsize=20):
    """Add a manual scalebar to a geographic coordinate plot"""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x_pos = xlim[0] + x_frac * (xlim[1] - xlim[0])
    y_pos = ylim[0] + y_frac * (ylim[1] - ylim[0])
    
    degrees_for_km = length_km / 102.0
    
    ax.plot([x_pos, x_pos + degrees_for_km], 
            [y_pos, y_pos], 
            color=color, linewidth=4, solid_capstyle='butt')
    
    ax.text(x_pos + degrees_for_km/2, 
            y_pos - 0.02,  
            f'{length_km} km', 
            ha='center', va='top',
            fontsize=fontsize,
            fontweight='bold',
            color=color)

def add_north_arrow(ax, x_frac=0.92, y_frac=0.90, length=0.17, color='black', fontsize=22):
    """Add a north arrow to a geographic coordinate plot"""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    x_pos = xlim[0] + x_frac * (xlim[1] - xlim[0])
    y_pos = ylim[0] + y_frac * (ylim[1] - ylim[0])
    
    # Draw arrow (bigger with increased linewidth)
    ax.annotate('', 
                xy=(x_pos, y_pos), 
                xytext=(x_pos, y_pos - length),
                arrowprops=dict(arrowstyle='->', color=color, linewidth=3.5, 
                               shrinkA=0, shrinkB=0))
    
    # Add 'N' text (bigger font)
    ax.text(x_pos, y_pos + length * 0.15, 'N', 
            ha='center', va='bottom',
            fontsize=fontsize,
            fontweight='bold',
            color=color)

def main():
    # --- Configuration ---
    SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]  # Auto-get script name
    
    PROJECT_ROOT = r"C:\Users\NagaiLab\GC_Project_work\submitted_RS_10_Bangla-crop_estimation"
    
    # --- Inputs ---
    SHAPEFILE_PATH = os.path.join(PROJECT_ROOT, "data", "shapefile", "BGD_adm", "BGD_adm2.shp")
    BOUNDARY_PATH = os.path.join(PROJECT_ROOT, "data", "shapefile", "bangladesh_boundary.geojson")
    CROPLAND_MASK_PATH = os.path.join(PROJECT_ROOT, "1__create_cropland_mask", "cropland_mask.tif")
    ELEVATION_MASK_PATH = os.path.join(PROJECT_ROOT, "2__prepare_dl_data_GPU-modified", "aligned_static", "bd_elevation.tif")

    # --- Outputs ---
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, SCRIPT_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_PLOT_PNG = os.path.join(OUTPUT_DIR, "cropland_elevation_masks.png")

    print("--- Plotting Cropland and Elevation Masks ---")
    print(f"Script name: {SCRIPT_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("NOTE: Creating plot with scale bars, grids, and north arrows...")

    # 1. Load Data
    print("Loading data...")
    try:
        districts_gdf = gpd.read_file(SHAPEFILE_PATH)
        boundary_gdf = gpd.read_file(BOUNDARY_PATH)
        
        # Check if raster files exist
        if not os.path.exists(CROPLAND_MASK_PATH):
            raise FileNotFoundError(f"Cropland mask not found: {CROPLAND_MASK_PATH}")
        if not os.path.exists(ELEVATION_MASK_PATH):
            raise FileNotFoundError(f"Elevation mask not found: {ELEVATION_MASK_PATH}")
        if not os.path.exists(BOUNDARY_PATH):
            raise FileNotFoundError(f"Boundary file not found: {BOUNDARY_PATH}")
            
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find input file. {e}")
        return

    # Check raster statistics
    print(f"\nChecking raster statistics...")
    
    # Cropland mask stats
    with rasterio.open(CROPLAND_MASK_PATH) as src:
        data = src.read(1)
        valid_data = data[data != src.nodata]
        print(f"  Cropland mask:")
        print(f"    Range: {valid_data.min():.4f} to {valid_data.max():.4f}")
        print(f"    Mean: {valid_data.mean():.4f}")
        print(f"    Unique values: {np.unique(valid_data)}")
    
    # Elevation mask stats
    with rasterio.open(ELEVATION_MASK_PATH) as src:
        data = src.read(1)
        valid_data = data[data != src.nodata]
        print(f"  Elevation mask (original):")
        print(f"    Range: {valid_data.min():.2f} to {valid_data.max():.2f} meters")
        print(f"    Mean: {valid_data.mean():.2f} meters")
        print(f"    95th percentile: {np.percentile(valid_data, 95):.2f} meters")

    # 2. Create plot with style from main comparison plot
    print("\nGenerating plot...")

    font_factor = 1.8
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24 * 1.2, 12 * 1.2))
    
    # Title
    fig.suptitle('Cropland Mask and Elevation Map of Bangladesh', 
                 fontsize=22 * font_factor, weight='bold', y=0.98)

    # --- Panel 1: Cropland Mask ---
    ax1.set_title('(a) Cropland Mask', fontsize=18 * font_factor)  # Changed A to a
    
    with rasterio.open(CROPLAND_MASK_PATH) as src:
        # Display the cropland mask
        # Use binary colormap since it's likely 0/1 data
        img1 = show(src, ax=ax1, cmap='Greens')
        
        # Overlay district boundaries in red
        districts_gdf.plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=0.7, alpha=0.8)
        
        # Create colorbar for the raster - show only 0 and 1
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.25)
        
        # Get the current image
        im = img1.get_images()[0]
        cbar = fig.colorbar(im, cax=cax, orientation='vertical', 
                           ticks=[0, 1])  # Only show 0 and 1
        # Moved "Cropland" label closer to colorbar (reduced labelpad from 15 to 8)
        cbar.set_label('Cropland', 
                       fontsize=16 * font_factor, labelpad=8)  # Changed from labelpad=15
        cbar.ax.tick_params(labelsize=14 * font_factor)
        cbar.ax.set_yticklabels(['0 (No)', '1 (Yes)'])  # Label the ticks

    # Add scalebar
    add_manual_scalebar(ax1, length_km=100, fontsize=20, y_frac=0.12)
    
    # Add north arrow
    add_north_arrow(ax1, fontsize=22, y_frac=0.90)

    ax1.set_xlabel("Longitude", fontsize=16 * font_factor)
    ax1.set_ylabel("Latitude", fontsize=16 * font_factor)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=14 * font_factor)
    ax1.set_aspect('equal', adjustable='box')

    # --- Panel 2: Elevation Mask ---
    ax2.set_title('(b) Elevation Map', fontsize=18 * font_factor)  # Changed B to b

    with rasterio.open(ELEVATION_MASK_PATH) as src:
        # First, mask the elevation data using the Bangladesh boundary
        print("Masking elevation data with Bangladesh boundary...")
        
        # Ensure the boundary is in the same CRS as the raster
        if boundary_gdf.crs != src.crs:
            boundary_gdf = boundary_gdf.to_crs(src.crs)
        
        # Get the geometry for masking
        geoms = boundary_gdf.geometry.values
        
        try:
            # Mask the raster with the boundary
            masked_data, masked_transform = mask(src, geoms, crop=True, filled=True)
            
            # Get the nodata value
            nodata = src.nodata
            
            # Create masked array for display
            if nodata is not None:
                display_data = np.ma.masked_equal(masked_data[0], nodata)
            else:
                display_data = masked_data[0]
            
            # Get the valid (unmasked) data
            valid_elevation_data = display_data.compressed()  # Get only non-masked values
            
            if len(valid_elevation_data) > 0:
                # Calculate vmin and vmax for colormap
                vmin = np.min(valid_elevation_data)
                vmax = min(np.max(valid_elevation_data), 110)  # Cap at 110 (just above 100)
                
                print(f"  Elevation mask (display range):")
                print(f"    vmin: {vmin:.2f} meters")
                print(f"    vmax: {vmax:.2f} meters (capped at just above 100)")
                print(f"    Actual max in data: {np.max(valid_elevation_data):.2f} meters")
                
            else:
                vmin = 0
                vmax = 100
                print("  No valid elevation data found within boundary")
            
            # Calculate extent from transform
            height, width = display_data.shape
            left = masked_transform[2]
            top = masked_transform[5]
            right = left + masked_transform[0] * width
            bottom = top + masked_transform[4] * height
            
            extent = [left, right, bottom, top]
            
            # Display the masked elevation data with stretched colormap
            img2 = ax2.imshow(display_data, 
                             extent=extent,
                             cmap='terrain',
                             vmin=vmin,
                             vmax=vmax)  # Stretched to just above 100
            
            # Overlay district boundaries in red
            districts_gdf.plot(ax=ax2, facecolor='none', edgecolor='red', linewidth=0.7, alpha=0.8)
            
            # Create colorbar for the raster
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.25)
            cbar = fig.colorbar(img2, cax=cax, orientation='vertical')
            # Moved "Elevation" label closer to colorbar (reduced labelpad from 15 to 8)
            cbar.set_label('Elevation (meters)', 
                           fontsize=16 * font_factor, labelpad=8)  # Changed from labelpad=15
            cbar.ax.tick_params(labelsize=14 * font_factor)
            
            # REMOVED the box saying colourmap lower right in plot (b)
            # Line was: ax2.text(0.98, 0.02, f'Colormap: {vmin:.0f}-{vmax:.0f} m', ...
            
        except Exception as e:
            print(f"Warning: Could not mask elevation data. Error: {e}")
            print("Falling back to standard display...")
            
            # Fallback to standard display if masking fails
            img2 = show(src, ax=ax2, cmap='terrain')
            districts_gdf.plot(ax=ax2, facecolor='none', edgecolor='red', linewidth=0.7, alpha=0.8)
            
            # Create colorbar for the raster
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.25)
            cbar = fig.colorbar(img2.get_images()[0], cax=cax, orientation='vertical')
            cbar.set_label('Elevation (meters)', 
                           fontsize=16 * font_factor, labelpad=8)  # Changed from labelpad=15
            cbar.ax.tick_params(labelsize=14 * font_factor)

    # Add scalebar
    add_manual_scalebar(ax2, length_km=100, fontsize=20, y_frac=0.12)
    
    # Add north arrow
    add_north_arrow(ax2, fontsize=22, y_frac=0.90)

    ax2.set_xlabel("Longitude", fontsize=16 * font_factor)
    ax2.set_ylabel("Latitude", fontsize=16 * font_factor)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=14 * font_factor)
    ax2.set_aspect('equal', adjustable='box')
    
    # Set consistent axis limits using shapefile bounds (using boundary instead)
    bounds = boundary_gdf.total_bounds
    ax1.set_xlim([bounds[0], bounds[2]])
    ax1.set_ylim([bounds[1], bounds[3]])
    ax2.set_xlim([bounds[0], bounds[2]])
    ax2.set_ylim([bounds[1], bounds[3]])

    # Save the plot
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    print(f"\nSaving plot to: {OUTPUT_PLOT_PNG}")
    plt.savefig(OUTPUT_PLOT_PNG, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Print summary
    print("\n" + "="*70)
    print("PLOT INTERPRETATION:")
    print("="*70)
    print("CROPLAND AND ELEVATION MASKS PLOT (cropland_elevation_masks.png):")
    print("  (a) Cropland Mask:")
    print("      - Shows areas with cropland (value = 1)")
    print("      - Non-cropland areas (value = 0)")
    print("      - Green color scheme for vegetation")
    print("      - Colorbar shows only 0 and 1")
    print("      - 'Cropland' label moved closer to colorbar")
    print("      - Red lines = District boundaries")
    print("")
    print("  (b) Elevation Map:")
    print("      - Shows elevation in meters above sea level")
    print("      - Terrain colormap stretched from minimum to just above 100 meters")
    print("      - Masked using Bangladesh boundary (no background outside country)")
    print("      - Areas above 100 meters shown with maximum colormap value")
    print("      - 'Elevation' label moved closer to colorbar")
    print("      - No additional text box in lower right")
    print("      - Red lines = District boundaries")
    print("")
    print("  Scale bars: 100 km reference for both maps")
    print("  North arrows: Top right of each subplot")
    print("  Grid lines: Latitude/Longitude grid")
    print("="*70)

    print(f"\n--- Script {SCRIPT_NAME}.py finished successfully. ---")

if __name__ == "__main__":
    main()