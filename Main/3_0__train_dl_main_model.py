import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, Dense, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import rasterio
import pandas as pd

# --- GPU Check ---
print("Checking for GPU devices...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs found.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU devices found. Model will train on CPU.")

# --- Configuration ---
# Get the script's own name without extension to use for the output folder
script_path = os.path.abspath(__file__)
script_name_with_ext = os.path.basename(script_path)
SCRIPT_NAME, _ = os.path.splitext(script_name_with_ext)

# Set PROJECT_ROOT to the base directory of the project
PROJECT_ROOT = "/srv/AB_YU/RS_10_Bangla-crop_estimation"

# --- Input Directories & Files ---
# Point to the final pre-processed data directory relative to PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "2.5__preprocess_for_training")

# --- Script Execution ---

# 1. Create output directory
print(f"Creating output directory: {SCRIPT_NAME}")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, SCRIPT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_MODEL_PATH = os.path.join(OUTPUT_DIR, "crop_vulnerability_model.keras")
OUTPUT_HISTORY_CSV = os.path.join(OUTPUT_DIR, "training_history.csv")
# Define paths for the new separate plots
OUTPUT_LOSS_PLOT = os.path.join(OUTPUT_DIR, "training_history_loss.png")
OUTPUT_METRICS_PLOT = os.path.join(OUTPUT_DIR, "training_history_metrics.png")


# 2. Load pre-processed data and metadata
print("Loading pre-processed data and metadata...")

# Load metadata to get shapes
metadata = np.load(os.path.join(DATA_DIR, "metadata.npy"), allow_pickle=True).item()
temporal_shape = metadata['temporal_shape']
static_shape = metadata['static_shape']
target_shape = metadata['target_shape']

# Load the memory-mapped arrays using the correct shape and dtype
temporal_features = np.memmap(os.path.join(DATA_DIR, "temporal_features.mmap"), dtype='float32', mode='r', shape=temporal_shape)
static_features = np.memmap(os.path.join(DATA_DIR, "static_features.mmap"), dtype='float32', mode='r', shape=static_shape)
y_target = np.memmap(os.path.join(DATA_DIR, "target.mmap"), dtype='float32', mode='r', shape=target_shape)

print("Successfully loaded pre-processed data.")
print(f"Loaded temporal_features shape: {temporal_features.shape}")
print(f"Loaded static_features shape: {static_features.shape}")
print(f"Loaded y_target shape: {y_target.shape}")

# --- Check Target Variable ---
print(f"\nTarget stats - Min: {y_target.min()}, Max: {y_target.max()}, Mean: {y_target.mean()}, Std: {y_target.std()}")
print(f"Number of zero values in target: {np.count_nonzero(y_target == 0)} / {len(y_target)}\n")

# Get dimensions for model input
num_pixels, num_years, num_temporal_features = temporal_features.shape
num_static_features = static_features.shape[1]

# 3. Spatially-Aware Data Splitting and Scaling
print("\n--- Spatially-Aware Data Splitting & Scaling ---")

# Load the coordinates of the valid pixels, saved by the preprocessing script
try:
    rows = np.load(os.path.join(DATA_DIR, "cropland_y_valid.npy"))
    cols = np.load(os.path.join(DATA_DIR, "cropland_x_valid.npy"))
    # Load original raster dimensions from the mask file
    with rasterio.open(os.path.join(PROJECT_ROOT, "1__create_cropland_mask", "cropland_mask.tif")) as src:
        height, width = src.height, src.width
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find coordinate or mask files needed for spatial split: {e}")
    print("Please ensure 'cropland_y_valid.npy', 'cropland_x_valid.npy' are in the DATA_DIR and the original cropland mask is accessible.")
    exit()

# Define the grid for spatial blocks (10x10 grid)
n_splits_y = 10
n_splits_x = 10
block_size_y = np.ceil(height / n_splits_y).astype(int)
block_size_x = np.ceil(width / n_splits_x).astype(int)

print(f"Dividing the {height}x{width} area into a {n_splits_y}x{n_splits_x} grid of spatial blocks.")

# Assign a unique block ID to each valid cropland pixel based on its coordinates
block_ids = (rows // block_size_y) * n_splits_x + (cols // block_size_x)
unique_block_ids = np.unique(block_ids)

print(f"Found {len(unique_block_ids)} unique spatial blocks containing valid cropland pixels.")

# Split the block IDs into training and validation sets
np.random.seed(42)
val_split_ratio = 0.2
val_block_ids = np.random.choice(
    unique_block_ids,
    size=int(len(unique_block_ids) * val_split_ratio),
    replace=False
)

# Create a boolean mask for all valid pixels: True if the pixel's block is in the validation set
is_val_pixel = np.isin(block_ids, val_block_ids)

# Get the indices for training and validation sets from the full array of valid pixels
indices = np.arange(num_pixels)
train_idx = indices[~is_val_pixel]
val_idx = indices[is_val_pixel]

print(f"Total valid pixels: {num_pixels}")
print(f"Training pixels (from {len(unique_block_ids) - len(val_block_ids)} blocks): {len(train_idx)}")
print(f"Validation pixels (from {len(val_block_ids)} blocks): {len(val_idx)}")

# --- Scale Features based on Spatial Split ---
print("\nScaling features based on spatially-split training data...")

# --- Scale Temporal Features ---
temporal_scaler = StandardScaler()
print("Reshaping and fitting temporal scaler...")
X_temp_train_slice_2d = temporal_features[train_idx].reshape(-1, num_temporal_features)
temporal_scaler.fit(X_temp_train_slice_2d)

print("Transforming all temporal data...")
temporal_features_scaled_2d = temporal_scaler.transform(temporal_features.reshape(-1, num_temporal_features))
temporal_features_scaled = temporal_features_scaled_2d.reshape(num_pixels, num_years, num_temporal_features)

# --- Scale Static Features ---
print("Fitting and transforming static scaler...")
static_scaler = StandardScaler()
static_scaler.fit(static_features[train_idx])
static_features_scaled = static_scaler.transform(static_features)

# --- Create final training and validation sets from scaled data ---
print("Creating final training and validation sets...")
X_temp_train = temporal_features_scaled[train_idx]
X_stat_train = static_features_scaled[train_idx]
y_train = y_target[train_idx]

X_temp_val = temporal_features_scaled[val_idx]
X_stat_val = static_features_scaled[val_idx]
y_val = y_target[val_idx]

print("Scaling and data splitting complete.")
print(f"Training temporal: {X_temp_train.shape}, Training static: {X_stat_train.shape}, Training target: {y_train.shape}")
print(f"Validation temporal: {X_temp_val.shape}, Validation static: {X_stat_val.shape}, Validation target: {y_val.shape}")

# 4. Define the Deep Learning Model
print("Defining the deep learning model...")

temporal_input = Input(shape=(num_years, num_temporal_features), name='temporal_input')
static_input = Input(shape=(num_static_features,), name='static_input')

temporal_features_gru = GRU(32, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', name='gru_layer')(temporal_input)
static_features_dense = Dense(16, activation='relu', kernel_initializer='he_normal', name='static_dense')(static_input)

fused = concatenate([temporal_features_gru, static_features_dense], name='fusion_layer')
hidden = Dense(32, activation='relu', kernel_initializer='he_normal', name='hidden_dense')(fused)
output = Dense(1, kernel_initializer='glorot_uniform', name='output')(hidden)

model = Model(inputs=[temporal_input, static_input], outputs=output)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# Add MAE and RMSE to the metrics to be tracked during training
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
model.summary()

# 5. Train the model
print("\nTraining the cropland-specific model...")

# Re-enabled EarlyStopping to get the best model and prevent overfitting.
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    min_delta=0.001
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7
)

history = model.fit(
    [X_temp_train, X_stat_train],
    y_train,
    validation_data=([X_temp_val, X_stat_val], y_val),
    epochs=20,
    batch_size=256,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler] # Re-added early_stopping
)

# 6. Final Model Evaluation
print("\n--- Final Model Evaluation (on Best Model) ---")
# Because restore_best_weights=True, the model object is already the best version.
final_metrics = model.evaluate([X_temp_val, X_stat_val], y_val, return_dict=True)
print("Final Validation Metrics:")
for name, value in final_metrics.items():
    print(f"  {name}: {value:.4f}")

# 7. Save the trained model and training history
print(f"\nSaving trained model to: {OUTPUT_MODEL_PATH}")
model.save(OUTPUT_MODEL_PATH)

print(f"Saving training history data to: {OUTPUT_HISTORY_CSV}")
history_df = pd.DataFrame(history.history)
history_df['epoch'] = history.epoch
history_df.to_csv(OUTPUT_HISTORY_CSV, index=False)

# 8. Create and save professional training history plots
print(f"Saving professional training history plots...")
plt.style.use('seaborn-v0_8-whitegrid')

# --- Plot 1: Loss (MSE) ---
fig1, ax1 = plt.subplots(figsize=(12, 8))
fig1.suptitle('Model Training History: Loss (MSE)', fontsize=20)

# Plotting data with markers and different line styles
ax1.plot(history_df.index, history_df['loss'], color='blue', linestyle='-', marker='o', markersize=4, label='Training Loss (MSE)')
ax1.plot(history_df.index, history_df['val_loss'], color='red', linestyle='--', marker='x', markersize=4, label='Validation Loss (MSE)')

# Find and annotate the best epoch
best_epoch = history_df['val_loss'].idxmin()
best_val_loss = history_df['val_loss'].min()
ax1.axvline(x=best_epoch, color='r', linestyle=':', label=f'Best Epoch: {best_epoch+1}')

# Increase font sizes
ax1.set_xlabel('Epoch', fontsize=16)
ax1.set_ylabel('Mean Squared Error', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.legend(fontsize=14)

print(f"Saving Loss plot to: {OUTPUT_LOSS_PLOT}")
fig1.savefig(OUTPUT_LOSS_PLOT, dpi=300, bbox_inches='tight')
plt.close(fig1)

# --- Plot 2: Other Metrics (MAE, RMSE) ---
fig2, ax2 = plt.subplots(figsize=(12, 8))
fig2.suptitle('Model Training History: Additional Metrics', fontsize=20)

ax2.plot(history_df.index, history_df['mae'], color='green', linestyle='-', marker='o', markersize=4, label='Training MAE')
ax2.plot(history_df.index, history_df['val_mae'], color='orange', linestyle='--', marker='x', markersize=4, label='Validation MAE')
ax2.plot(history_df.index, history_df['rmse'], color='purple', linestyle='-', marker='s', markersize=4, label='Training RMSE')
ax2.plot(history_df.index, history_df['val_rmse'], color='brown', linestyle='--', marker='^', markersize=4, label='Validation RMSE')
ax2.axvline(x=best_epoch, color='r', linestyle=':')

# Increase font sizes
ax2.set_xlabel('Epoch', fontsize=16)
ax2.set_ylabel('Metric Value', fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.legend(fontsize=14)

print(f"Saving Metrics plot to: {OUTPUT_METRICS_PLOT}")
fig2.savefig(OUTPUT_METRICS_PLOT, dpi=300, bbox_inches='tight')
plt.close(fig2)


print(f"\nScript {SCRIPT_NAME}.py finished successfully.")