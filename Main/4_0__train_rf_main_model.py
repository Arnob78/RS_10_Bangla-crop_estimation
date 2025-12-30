import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
import matplotlib.pyplot as plt
import joblib
import rasterio
from scipy.stats import pearsonr

# --- Configuration ---
script_path = os.path.abspath(__file__)
script_filename = os.path.basename(script_path)
SCRIPT_NAME, _ = os.path.splitext(script_filename)

print(f"Script name detected: {SCRIPT_NAME}")

# Set PROJECT_ROOT to the base directory of the project
PROJECT_ROOT = "/srv/AB_YU/RS_10_Bangla-crop_estimation"
DATA_DIR = os.path.join(PROJECT_ROOT, "2.5__preprocess_for_training")

# --- Output Directory ---
print(f"Creating output directory: {SCRIPT_NAME}")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, SCRIPT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_MODEL_PATH = os.path.join(OUTPUT_DIR, "random_forest_model.joblib")
OUTPUT_METRICS_CSV = os.path.join(OUTPUT_DIR, "rf_evaluation_metrics.csv")
OUTPUT_REPORT_TXT = os.path.join(OUTPUT_DIR, "rf_evaluation_report.txt")
OUTPUT_IMPORTANCE_PLOT = os.path.join(OUTPUT_DIR, "rf_feature_importance.png")
OUTPUT_SCATTER_PLOT = os.path.join(OUTPUT_DIR, "rf_predictions_vs_actual.png")

def add_value_labels_inside(ax, bars, fontsize=13.5, fmt='.5f', padding_percent=2):  # Changed to .5f for more decimals
    """Add value labels inside the bars (at the end of each bar)"""
    # Get the current x-axis limits
    xlim = ax.get_xlim()
    max_x = xlim[1]
    
    for bar in bars:
        width = bar.get_width()
        # Position label at the end of the bar, but slightly inside
        label_x_pos = width - (max_x * padding_percent / 100)
        
        # Only add label if there's enough space inside the bar
        if label_x_pos > width * 0.1:  # Ensure label is not too close to start
            ax.text(label_x_pos, 
                   bar.get_y() + bar.get_height()/2,
                   f'{width:{fmt}}',
                   ha='right', 
                   va='center',
                   fontsize=fontsize,  # Increased font size
                   fontweight='normal',
                   color='white')  # White text for contrast inside bars

def main():
    # 1. Load metadata and recreate train/val split
    print("Loading metadata and recreating train/val split...")
    try:
        metadata = np.load(os.path.join(DATA_DIR, "metadata.npy"), allow_pickle=True).item()
    except FileNotFoundError:
        print("ERROR: metadata.npy not found!")
        print(f"Looking in: {DATA_DIR}")
        return
    
    temporal_shape = metadata['temporal_shape']
    static_shape = metadata['static_shape']
    target_shape = metadata['target_shape']

    num_pixels, num_years, num_temporal_features = temporal_shape
    num_static_features = static_shape[1]

    print(f"Dataset shapes:")
    print(f"  Temporal: {temporal_shape}")
    print(f"  Static: {static_shape}")
    print(f"  Target: {target_shape}")
    print(f"  Total features after flattening: {num_years * num_temporal_features + num_static_features}")

    # 2. Recreate spatial split
    print("\nRecreating spatial train/val split...")
    try:
        rows = np.load(os.path.join(DATA_DIR, "cropland_y_valid.npy"))
        cols = np.load(os.path.join(DATA_DIR, "cropland_x_valid.npy"))
        print(f"Loaded coordinates - rows: {rows.shape}, cols: {cols.shape}")
        
        with rasterio.open(os.path.join(PROJECT_ROOT, "1__create_cropland_mask", "cropland_mask.tif")) as src:
            height, width = src.height, src.width
            print(f"Mask dimensions - height: {height}, width: {width}")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}")
        return

    n_splits_y, n_splits_x = 10, 10
    block_size_y = np.ceil(height / n_splits_y).astype(int)
    block_size_x = np.ceil(width / n_splits_x).astype(int)
    block_ids = (rows // block_size_y) * n_splits_x + (cols // block_size_x)
    
    np.random.seed(42)
    unique_block_ids = np.unique(block_ids)
    val_split_ratio = 0.2
    val_block_ids = np.random.choice(unique_block_ids, size=int(len(unique_block_ids) * val_split_ratio), replace=False)
    is_val_pixel = np.isin(block_ids, val_block_ids)
    indices = np.arange(num_pixels)
    train_idx = indices[~is_val_pixel]
    val_idx = indices[is_val_pixel]

    print(f"Training samples: {len(train_idx):,}")
    print(f"Validation samples: {len(val_idx):,}")

    # 3. Load and Prepare Data for Random Forest
    print("\nLoading and preparing data for Random Forest...")
    print("WARNING: This step loads the entire dataset into memory and may require significant RAM!")
    
    # Check if memory-mapped files exist
    temporal_mmap_path = os.path.join(DATA_DIR, "temporal_features.mmap")
    static_mmap_path = os.path.join(DATA_DIR, "static_features.mmap")
    target_mmap_path = os.path.join(DATA_DIR, "target.mmap")
    
    for path in [temporal_mmap_path, static_mmap_path, target_mmap_path]:
        if not os.path.exists(path):
            print(f"ERROR: Missing file: {path}")
            return
    
    try:
        # Load data using memmap
        temporal_features = np.memmap(temporal_mmap_path, dtype='float32', mode='r', shape=temporal_shape)
        static_features = np.memmap(static_mmap_path, dtype='float32', mode='r', shape=static_shape)
        y_target = np.memmap(target_mmap_path, dtype='float32', mode='r', shape=target_shape)
        
        # Calculate total memory needed
        temporal_memory = temporal_features.size * 4 / (1024**3)  # GB
        static_memory = static_features.size * 4 / (1024**3)
        target_memory = y_target.size * 4 / (1024**3)
        print(f"Estimated memory usage:")
        print(f"  Temporal features: {temporal_memory:.2f} GB")
        print(f"  Static features: {static_memory:.2f} GB")
        print(f"  Target: {target_memory:.2f} GB")
        print(f"  Total: {temporal_memory + static_memory + target_memory:.2f} GB")
        
        # Take smaller sample if needed (for testing/debugging)
        # Uncomment these lines if memory is an issue
        # sample_size = 500000  # Adjust based on available memory
        # train_idx = train_idx[:sample_size]
        # val_idx = val_idx[:min(100000, len(val_idx))]
        
        # Flatten the temporal data
        print("Flattening temporal data...")
        X_temporal_flat = temporal_features.reshape(num_pixels, -1)
        
        # Combine temporal and static features
        print("Combining features...")
        X_full = np.concatenate([X_temporal_flat, static_features], axis=1)
        
        # Split into training and validation sets
        print("Creating train/validation splits...")
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_target[train_idx].ravel(), y_target[val_idx].ravel()

        print(f"Train features shape: {X_train.shape}")
        print(f"Validation features shape: {X_val.shape}")
        print(f"Train targets shape: {y_train.shape}")
        print(f"Validation targets shape: {y_val.shape}")
        
        # Check for NaN/inf values
        print(f"NaN in X_train: {np.isnan(X_train).sum()}")
        print(f"NaN in y_train: {np.isnan(y_train).sum()}")
        print(f"NaN in X_val: {np.isnan(X_val).sum()}")
        print(f"NaN in y_val: {np.isnan(y_val).sum()}")

    except MemoryError:
        print("\nFATAL ERROR: MemoryError occurred while loading data.")
        print("Try using a smaller sample or running on a machine with more RAM.")
        print("You can uncomment the sample_size lines above to reduce memory usage.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Train Random Forest Model
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60)
    
    # Random Forest parameters
    n_estimators = 100  # Number of trees
    max_samples = 100000  # Maximum samples per tree (reduce if memory is issue)
    
    print(f"Training Random Forest with {n_estimators} trees...")
    print(f"Using max_samples = {max_samples:,} per tree")
    
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=42,
        n_jobs=-1,  # Use all available cores
        oob_score=True,  # Out-of-bag score for quick validation
        verbose=2,  # Show progress
        max_features='sqrt',  # Number of features to consider for best split
        min_samples_split=5,  # Minimum samples required to split a node
        min_samples_leaf=2,  # Minimum samples required at a leaf node
    )
    
    print("Starting training (this may take a while)...")
    rf.fit(X_train, y_train)
    
    print(f"\nModel training complete!")
    print(f"OOB Score (Out-of-Bag R²): {rf.oob_score_:.4f}")

    # 5. Evaluate the model
    print("\n" + "="*60)
    print("EVALUATING RANDOM FOREST MODEL")
    print("="*60)
    
    print("Making predictions on validation set...")
    y_pred = rf.predict(X_val)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    explained_variance = explained_variance_score(y_val, y_pred)
    correlation, p_value = pearsonr(y_val, y_pred)
    bias = np.mean(y_pred - y_val)
    
    # Calculate relative errors
    mean_target = np.abs(y_val).mean()
    mae_percent = (mae / mean_target) * 100 if mean_target != 0 else float('inf')
    rmse_percent = (rmse / mean_target) * 100 if mean_target != 0 else float('inf')

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Explained_Variance': explained_variance,
        'Pearson_Correlation': correlation,
        'Correlation_p_value': p_value,
        'Bias': bias,
        'MAE_percent': mae_percent,
        'RMSE_percent': rmse_percent,
        'OOB_Score': rf.oob_score_,
        'Validation_samples': len(y_val)
    }

    print("\nPERFORMANCE METRICS:")
    print(f"  R-squared (R²): {r2:.4f}")
    print(f"  OOB Score: {rf.oob_score_:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Explained Variance: {explained_variance:.4f}")
    print(f"  Correlation: {correlation:.4f} (p-value: {p_value:.2e})")
    print(f"  Bias: {bias:.6f}")
    print(f"  MAE as % of mean target: {mae_percent:.2f}%")
    print(f"  RMSE as % of mean target: {rmse_percent:.2f}%")

    # 6. Save Metrics and Report
    print("\nSaving metrics and report...")
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    metrics_df.to_csv(OUTPUT_METRICS_CSV, index=False)
    
    with open(OUTPUT_REPORT_TXT, 'w') as f:
        f.write("="*60 + "\n")
        f.write("RANDOM FOREST BASELINE EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. MODEL INFORMATION\n")
        f.write(f"   Model: Random Forest Regressor\n")
        f.write(f"   Trees: {n_estimators}\n")
        f.write(f"   Max samples per tree: {max_samples:,}\n")
        f.write(f"   Total features: {X_train.shape[1]}\n")
        f.write(f"   Training samples: {len(train_idx):,}\n")
        f.write(f"   Validation samples: {len(val_idx):,}\n\n")
        
        f.write("2. PERFORMANCE METRICS\n")
        f.write("="*40 + "\n")
        for name, value in metrics.items():
            if 'p_value' in name:
                f.write(f"{name}: {value:.2e}\n")
            elif 'percent' in name:
                f.write(f"{name}: {value:.2f}%\n")
            elif name in ['R2', 'Explained_Variance', 'Pearson_Correlation', 'OOB_Score']:
                f.write(f"{name}: {value:.4f}\n")
            else:
                f.write(f"{name}: {value:.6f}\n")
        
        f.write("\n3. INTERPRETATION\n")
        f.write("="*40 + "\n")
        if r2 > 0.7:
            f.write("✓ Excellent performance (R² > 0.7)\n")
        elif r2 > 0.5:
            f.write("✓ Good performance (R² > 0.5)\n")
        elif r2 > 0.3:
            f.write("✓ Moderate performance (R² > 0.3)\n")
        else:
            f.write("⚠ Performance needs improvement (R² ≤ 0.3)\n")
    
    print(f"Metrics saved to {OUTPUT_METRICS_CSV}")
    print(f"Report saved to {OUTPUT_REPORT_TXT}")

    # 7. Feature Importances
    print("\nCalculating feature importances...")
    importances = rf.feature_importances_
    
    # Create feature names with corrected logic based on data creation script 2.5
    print("Generating corrected feature names...")
    temporal_feature_names = []
    
    # Data is structured as Year0(all 60 bands), Year1(all 60 bands), etc.
    # The 60 bands per year are composed of 12 NDVI bands, then 48 climate bands.
    # The climate bands are themselves ordered by variable.
    
    climate_vars_in_order = ['precip', 'tmin', 'tmax', 'vpd']
    for yr_idx in range(num_years): # 0 to 21
        # Add 12 NDVI feature names for the current year
        for month_idx in range(1, 13):
             temporal_feature_names.append(f"yr_{yr_idx}_NDVI_month_{month_idx}")
        # Add 48 climate feature names for the current year
        for var in climate_vars_in_order:
             for month_idx in range(1, 13):
                 temporal_feature_names.append(f"yr_{yr_idx}_{var}_month_{month_idx}")

    # Static features are appended after all temporal features.
    # From analysis of 2.5__*.py, 'aspect' is replaced by sin/cos, resulting in 7 static features.
    static_feature_names = ['aspect_sin', 'aspect_cos', 'elevation', 'slope', 'soil_clay', 'soil_ph', 'soil_sand']
    
    all_feature_names = temporal_feature_names + static_feature_names
    
    # Verification check
    if len(all_feature_names) != X_train.shape[1]:
        print(f"FATAL ERROR: Number of generated feature names ({len(all_feature_names)}) does not match number of data columns ({X_train.shape[1]}).")
        return
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Save importance to CSV
    importance_csv = os.path.join(OUTPUT_DIR, "rf_feature_importance.csv")
    importance_df.to_csv(importance_csv, index=False)
    print(f"Feature importance saved to {importance_csv}")
    
    # Plot top 30 features with value labels (using original outside labels for main plot)
    print("Creating feature importance plot...")
    plt.figure(figsize=(16, 12))  # Increased figure size for larger fonts
    top_features = importance_df.head(30)
    bars = plt.barh(top_features['feature'], top_features['importance'])
    plt.gca().invert_yaxis()
    plt.title('Top 30 Feature Importances - Random Forest', fontsize=24, weight='bold')  # Increased from 16
    plt.xlabel('Importance', fontsize=21)  # Increased from 14
    
    # Increase tick label font size
    plt.tick_params(axis='both', which='major', labelsize=18)  # Increased tick labels
    
    # Add value labels to bars (outside) with larger font
    for bar in bars:
        width = bar.get_width()
        plt.text(width * 1.01, bar.get_y() + bar.get_height()/2,
                f'{width:.5f}',  # Changed to 5 decimal places
                ha='left', va='center',
                fontsize=13.5)  # Increased from 9
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMPORTANCE_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {OUTPUT_IMPORTANCE_PLOT}")
    
    # Plot top static features with value labels INSIDE bars
    static_importance_df = importance_df[importance_df['feature'].isin(static_feature_names)].copy()
    if len(static_importance_df) > 0:
        print("Creating static feature importance plot with labels inside bars...")
        plt.figure(figsize=(14, 10))  # Increased figure size
        bars = plt.barh(static_importance_df['feature'], static_importance_df['importance'])
        plt.gca().invert_yaxis()
        plt.title('Static Feature Importances - Random Forest', fontsize=21, weight='bold')  # Increased from 14
        plt.xlabel('Importance', fontsize=18)  # Increased from 12
        
        # Increase tick label font size
        plt.tick_params(axis='both', which='major', labelsize=15)
        
        # Add value labels INSIDE the bars with larger font
        # Note: fmt is already set to '.5f' in the function definition
        add_value_labels_inside(plt.gca(), bars, fontsize=15, fmt='.5f', padding_percent=2)  # Changed to 5 decimal places
        
        plt.tight_layout()
        static_plot_path = os.path.join(OUTPUT_DIR, "rf_static_feature_importance.png")
        plt.savefig(static_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Static feature importance plot saved to {static_plot_path}")

    # 8. Create scatter plot of predictions vs actual with metrics in LOWER RIGHT
    print("\nCreating predictions vs actual scatter plot...")
    plt.figure(figsize=(12, 10))  # Increased figure size
    plt.scatter(y_val, y_pred, alpha=0.3, s=10, color='blue', edgecolors='black', linewidths=0.5)
    
    # Add 1:1 line
    min_val = min(y_val.min(), y_pred.min())
    max_val = max(y_val.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')
    
    # Add regression line
    z = np.polyfit(y_val, y_pred, 1)
    p = np.poly1d(z)
    plt.plot([min_val, max_val], [p(min_val), p(max_val)], 'g-', linewidth=2, 
             label=f'Regression: y={z[0]:.3f}x+{z[1]:.3f}')
    
    plt.xlabel('Actual Values', fontsize=21)  # Increased from 14
    plt.ylabel('Predicted Values', fontsize=21)  # Increased from 14
    plt.title(f'Random Forest: Predictions vs Actual', fontsize=24, weight='bold')  # Increased from 16
    
    # Increase tick label font size
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    # Move legend to upper left (more central) and make it larger
    plt.legend(fontsize=18, loc='upper left')  # Changed back to upper left
    
    plt.grid(True, alpha=0.3)
    
    # Add text box with metrics in LOWER RIGHT corner with larger font
    textstr = '\n'.join((
        f'R² = {r2:.4f}',
        f'RMSE = {rmse:.4f}',
        f'MAE = {mae:.4f}',
        f'Corr = {correlation:.4f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    
    # Position in lower right (x=0.95, y=0.05)
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=16.5,  # Increased from 11
             verticalalignment='bottom', horizontalalignment='right',
             bbox=props)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_SCATTER_PLOT, dpi=300)
    plt.close()
    print(f"Scatter plot saved to {OUTPUT_SCATTER_PLOT}")

    # 9. Save the trained model
    print(f"\nSaving trained model to {OUTPUT_MODEL_PATH}")
    joblib.dump(rf, OUTPUT_MODEL_PATH)
    
    print(f"\n" + "="*60)
    print("SCRIPT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nAll results saved in: {OUTPUT_DIR}")
    print(f"Model file: {OUTPUT_MODEL_PATH}")
    print(f"Metrics file: {OUTPUT_METRICS_CSV}")
    print(f"Report file: {OUTPUT_REPORT_TXT}")

if __name__ == "__main__":
    main()