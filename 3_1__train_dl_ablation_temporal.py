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

# Set PROJECT_ROOT to the base directory of the project on the server
PROJECT_ROOT = "/srv/AB_YU/RS_10_Bangla-crop_estimation"

# --- Input Directories & Files ---
# Point to the final pre-processed data directory
DATA_DIR = os.path.join(PROJECT_ROOT, "2.5__preprocess_for_training")

# Define all input file paths
METADATA_PATH = os.path.join(DATA_DIR, "metadata.npy")
TEMPORAL_FEATURES_PATH = os.path.join(DATA_DIR, "temporal_features.mmap")
STATIC_FEATURES_PATH = os.path.join(DATA_DIR, "static_features.mmap")
TARGET_PATH = os.path.join(DATA_DIR, "target.mmap")
CROPLAND_Y_VALID_PATH = os.path.join(DATA_DIR, "cropland_y_valid.npy")
CROPLAND_X_VALID_PATH = os.path.join(DATA_DIR, "cropland_x_valid.npy")
CROPLAND_MASK_PATH = os.path.join(PROJECT_ROOT, "1__create_cropland_mask", "cropland_mask.tif")

# --- Script Execution ---

# 1. Create output directory in the project root
print(f"Creating output directory: {SCRIPT_NAME}")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, SCRIPT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file paths
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_DIR, "ablation_temporal_only_model.keras")
OUTPUT_HISTORY_CSV = os.path.join(OUTPUT_DIR, "training_history_temporal_only.csv")
OUTPUT_LOSS_PLOT = os.path.join(OUTPUT_DIR, "training_history_loss_temporal_only.png")
OUTPUT_METRICS_PLOT = os.path.join(OUTPUT_DIR, "training_history_metrics_temporal_only.png")

# 2. Load pre-processed data and metadata
print("Loading pre-processed data and metadata...")
print(f"Metadata: {METADATA_PATH}")
print(f"Temporal features: {TEMPORAL_FEATURES_PATH}")
print(f"Static features: {STATIC_FEATURES_PATH}")
print(f"Target: {TARGET_PATH}")

# Load metadata to get shapes
try:
    metadata = np.load(METADATA_PATH, allow_pickle=True).item()
    temporal_shape = metadata['temporal_shape']
    target_shape = metadata['target_shape']
    print("Successfully loaded metadata.")
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find metadata file: {e}")
    exit()

# Load the memory-mapped arrays using the correct shape and dtype
try:
    temporal_features = np.memmap(TEMPORAL_FEATURES_PATH, dtype='float32', mode='r', shape=temporal_shape)
    y_target = np.memmap(TARGET_PATH, dtype='float32', mode='r', shape=target_shape)
    print("Successfully loaded memory-mapped data.")
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find memory-mapped data files: {e}")
    exit()

print(f"Loaded temporal_features shape: {temporal_features.shape}")
print(f"Loaded y_target shape: {y_target.shape}")

# --- Check Target Variable ---
print(f"\nTarget stats - Min: {y_target.min()}, Max: {y_target.max()}, Mean: {y_target.mean()}, Std: {y_target.std()}")
print(f"Number of zero values in target: {np.count_nonzero(y_target == 0)} / {len(y_target)}\n")

# Get dimensions for model input
num_pixels, num_years, num_temporal_features = temporal_features.shape

# 3. Spatially-Aware Data Splitting and Scaling
print("\n--- Spatially-Aware Data Splitting & Scaling ---")

# Load the coordinates of the valid pixels
try:
    rows = np.load(CROPLAND_Y_VALID_PATH)
    cols = np.load(CROPLAND_X_VALID_PATH)
    print(f"Loaded coordinate files: {len(rows)} valid pixels")
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find coordinate files: {e}")
    exit()

# Load original raster dimensions from the mask file
try:
    with rasterio.open(CROPLAND_MASK_PATH) as src:
        height, width = src.height, src.width
    print(f"Loaded mask file: {height}x{width} dimensions")
except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find cropland mask file: {e}")
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
print("\nScaling temporal features based on spatially-split training data...")

# --- Scale Temporal Features ---
temporal_scaler = StandardScaler()
print("Reshaping and fitting temporal scaler...")
X_temp_train_slice_2d = temporal_features[train_idx].reshape(-1, num_temporal_features)
temporal_scaler.fit(X_temp_train_slice_2d)

print("Transforming all temporal data...")
temporal_features_scaled_2d = temporal_scaler.transform(temporal_features.reshape(-1, num_temporal_features))
temporal_features_scaled = temporal_features_scaled_2d.reshape(num_pixels, num_years, num_temporal_features)

# --- Create final training and validation sets from scaled data ---
print("Creating final training and validation sets...")
X_temp_train = temporal_features_scaled[train_idx]
y_train = y_target[train_idx]

X_temp_val = temporal_features_scaled[val_idx]
y_val = y_target[val_idx]

print("Scaling and data splitting complete.")
print(f"Training temporal: {X_temp_train.shape}, Training target: {y_train.shape}")
print(f"Validation temporal: {X_temp_val.shape}, Validation target: {y_val.shape}")

# 4. Define the Deep Learning Model (Temporal-Only Ablation)
print("Defining the temporal-only ablation model...")

temporal_input = Input(shape=(num_years, num_temporal_features), name='temporal_input')

temporal_features_gru = GRU(32, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', name='gru_layer')(temporal_input)

# No concatenation, directly pass GRU output to dense layers
hidden = Dense(32, activation='relu', kernel_initializer='he_normal', name='hidden_dense')(temporal_features_gru)
output = Dense(1, kernel_initializer='glorot_uniform', name='output')(hidden)

model = Model(inputs=temporal_input, outputs=output) # Only temporal input

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
model.summary()

# 5. Train the model
print("\nTraining the temporal-only ablation model...")

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
    X_temp_train, # Only temporal input
    y_train,
    validation_data=(X_temp_val, y_val), # Only temporal input
    epochs=20,
    batch_size=256,
    verbose=1,
    callbacks=[early_stopping, lr_scheduler]
)

# 6. Final Model Evaluation
print("\n--- Final Model Evaluation (on Best Temporal-Only Model) ---")
final_metrics = model.evaluate(X_temp_val, y_val, return_dict=True) # Only temporal input
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
fig1.suptitle('Temporal-Only Ablation Model Training History: Loss (MSE)', fontsize=20)

ax1.plot(history_df.index, history_df['loss'], color='blue', linestyle='-', marker='o', markersize=4, label='Training Loss (MSE)')
ax1.plot(history_df.index, history_df['val_loss'], color='red', linestyle='--', marker='x', markersize=4, label='Validation Loss (MSE)')

best_epoch = history_df['val_loss'].idxmin()
best_val_loss = history_df['val_loss'].min()
ax1.axvline(x=best_epoch, color='r', linestyle=':', label=f'Best Epoch: {best_epoch+1}')

ax1.set_xlabel('Epoch', fontsize=16)
ax1.set_ylabel('Mean Squared Error', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.legend(fontsize=14)

print(f"Saving Loss plot to: {OUTPUT_LOSS_PLOT}")
fig1.savefig(OUTPUT_LOSS_PLOT, dpi=300, bbox_inches='tight')
plt.close(fig1)

# --- Plot 2: Other Metrics (MAE, RMSE) ---
fig2, ax2 = plt.subplots(figsize=(12, 8))
fig2.suptitle('Temporal-Only Ablation Model Training History: Additional Metrics', fontsize=20)

ax2.plot(history_df.index, history_df['mae'], color='green', linestyle='-', marker='o', markersize=4, label='Training MAE')
ax2.plot(history_df.index, history_df['val_mae'], color='orange', linestyle='--', marker='x', markersize=4, label='Validation MAE')
ax2.plot(history_df.index, history_df['rmse'], color='purple', linestyle='-', marker='s', markersize=4, label='Training RMSE')
ax2.plot(history_df.index, history_df['val_rmse'], color='brown', linestyle='--', marker='^', markersize=4, label='Validation RMSE')
ax2.axvline(x=best_epoch, color='r', linestyle=':')

ax2.set_xlabel('Epoch', fontsize=16)
ax2.set_ylabel('Metric Value', fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.legend(fontsize=14)

print(f"Saving Metrics plot to: {OUTPUT_METRICS_PLOT}")
fig2.savefig(OUTPUT_METRICS_PLOT, dpi=300, bbox_inches='tight')
plt.close(fig2)

print(f"\nScript {SCRIPT_NAME}.py finished successfully.")
print(f"All outputs saved to: {OUTPUT_DIR}")