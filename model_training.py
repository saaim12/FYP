import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import json
# Import MinMaxScaler for data scaling
from sklearn.preprocessing import MinMaxScaler
import shutil # For cleaning up directories

# --- Configuration ---
NUM_SAMPLES = 5000       # Number of training samples
EPOCHS = 100             # Increased training epochs
BATCH_SIZE = 32          # Training batch size
TEMP_MIN = 0.0           # Minimum temperature (°C)
TEMP_MAX = 50.0          # Maximum temperature (°C)
LATENT_DIM = 2           # Dimension of the latent space (encoded representation)
NOISE_FACTOR = 0.5       # Add some noise to simulate real-world fluctuations

# --- Output Directories ---
EXPORT_DIR = 'exported_models'
TFLITE_EXPORT_DIR = os.path.join(EXPORT_DIR, 'tflite')
TFJS_EXPORT_DIR = os.path.join(EXPORT_DIR, 'tfjs')
KERAS_TEMP_DIR = os.path.join(EXPORT_DIR, 'temp_keras') # Temporary dir for Keras model

os.makedirs(TFLITE_EXPORT_DIR, exist_ok=True)
os.makedirs(TFJS_EXPORT_DIR, exist_ok=True)
os.makedirs(KERAS_TEMP_DIR, exist_ok=True) # Create temp Keras dir

print(f"TensorFlow Version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# --- 1. Generate Synthetic Temperature Data ---
# Simulate realistic temperature fluctuations (e.g., using a sine wave + noise)
time = np.linspace(0, 100, NUM_SAMPLES)
base_temp = TEMP_MIN + (TEMP_MAX - TEMP_MIN) / 2 * (1 + np.sin(time * 0.1))
noise = np.random.normal(0, NOISE_FACTOR, NUM_SAMPLES)
temperature_data = base_temp + noise
temperature_data = np.clip(temperature_data, TEMP_MIN, TEMP_MAX)
temperature_data = temperature_data.astype(np.float32).reshape(-1, 1)

print(f"Generated {NUM_SAMPLES} temperature samples.")
print(f"Original data shape: {temperature_data.shape}")
print(f"Example original data points: {temperature_data[:5].flatten()}")

# --- NEW: Scale Data to [0, 1] ---
scaler = MinMaxScaler()
# Fit and transform the data
# In a real scenario with separate train/test sets, fit only on training data
scaled_temperature_data = scaler.fit_transform(temperature_data)

print(f"Scaled data shape: {scaled_temperature_data.shape}")
print(f"Example scaled data points: {scaled_temperature_data[:5].flatten()}")


# Visualize the generated data (Optional - showing original scale)
plt.figure(figsize=(10, 4))
plt.plot(time, temperature_data.flatten(), label='Simulated Temperature (°C)')
plt.title('Generated DHT11 Temperature Data (Original Scale)')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Temperature (°C)')
plt.ylim(TEMP_MIN - 5, TEMP_MAX + 5)
plt.legend()
plt.grid(True)
plt.show()

# --- 2. Define Autoencoder Model Architecture ---
# Using the scaled data shape
input_dim = scaled_temperature_data.shape[1] # Should still be 1

# --- Encoder ---
# Slightly increased complexity
encoder_input = keras.Input(shape=(input_dim,), name='encoder_input')
x = layers.Dense(16, activation='relu')(encoder_input) # Increased neurons
x = layers.Dense(8, activation='relu')(x)             # Increased neurons
encoder_output = layers.Dense(LATENT_DIM, activation='linear', name='encoder_output')(x)
encoder = keras.Model(encoder_input, encoder_output, name='encoder')
print("\n--- Encoder Summary ---")
encoder.summary()

# --- Decoder ---
decoder_input = keras.Input(shape=(LATENT_DIM,), name='decoder_input')
x = layers.Dense(8, activation='relu')(decoder_input)   # Increased neurons
x = layers.Dense(16, activation='relu')(x)            # Increased neurons
# Output activation is linear for regression, but sigmoid might be better
# since we scaled data to [0,1]. Let's try sigmoid.
decoder_output = layers.Dense(input_dim, activation='sigmoid', name='decoder_output')(x)
decoder = keras.Model(decoder_input, decoder_output, name='decoder')
print("\n--- Decoder Summary ---")
decoder.summary()

# --- Autoencoder (End-to-End) ---
autoencoder_input = keras.Input(shape=(input_dim,), name='autoencoder_input')
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = keras.Model(autoencoder_input, decoded, name='autoencoder')
print("\n--- Autoencoder Summary ---")
autoencoder.summary()

# --- 3. Compile and Train the Autoencoder ---
# Compile with Adam optimizer and Mean Squared Error loss
autoencoder.compile(optimizer='adam', loss='mse')

print("\n--- Training Autoencoder on SCALED data ---")
history = autoencoder.fit(
    scaled_temperature_data,
    scaled_temperature_data, # Train to reconstruct the SCALED input
    epochs=EPOCHS,           # Using increased epochs
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.2
)

# Plot training & validation loss values
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training (Scaled Data)')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# --- 4. Test Reconstruction ---
print("\n--- Reconstruction Test (Scaled vs Original) ---")
num_test_samples = 10
test_indices = np.random.choice(len(scaled_temperature_data), num_test_samples, replace=False)

# Get original and scaled samples for comparison
original_samples_unscaled = temperature_data[test_indices]
original_samples_scaled = scaled_temperature_data[test_indices]

# Predict using the autoencoder (works on scaled data)
reconstructed_samples_scaled = autoencoder.predict(original_samples_scaled)

# --- NEW: Inverse transform the reconstructed data to original scale ---
reconstructed_samples_unscaled = scaler.inverse_transform(reconstructed_samples_scaled)

print("Original Temp (°C) | Reconstructed Temp (°C) | Difference (°C)")
print("-" * 60)
total_diff = 0
for i in range(num_test_samples):
    orig_unscaled = original_samples_unscaled[i, 0]
    recon_unscaled = reconstructed_samples_unscaled[i, 0]
    diff = abs(orig_unscaled - recon_unscaled)
    total_diff += diff
    print(f"{orig_unscaled:18.2f} | {recon_unscaled:25.2f} | {diff:15.2f}")
print("-" * 60)
print(f"Average Reconstruction Difference: {total_diff / num_test_samples:.2f}°C")


# --- 5. Export Encoder Model to TensorFlow Lite ---
# NOTE: The TFLite model will expect SCALED input [0, 1]
print("\n--- Exporting Encoder to TFLite (expects scaled input [0,1]) ---")
converter = tf.lite.TFLiteConverter.from_keras_model(encoder)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
# Ensure float32 input/output as before
# converter.inference_input_type = tf.float32 # Usually inferred
# converter.inference_output_type = tf.float32 # Usually inferred

tflite_model = converter.convert()

tflite_model_path = os.path.join(TFLITE_EXPORT_DIR, 'encoder_model.tflite')
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
print(f"Encoder TFLite model saved to: {tflite_model_path}")

# --- Convert TFLite model to C array ---
c_array_path = os.path.join(TFLITE_EXPORT_DIR, 'encoder_model_data.h')
c_var_name = 'g_encoder_model_data'
exit_code_xxd = -1 # Initialize exit code
try:
    command = f"xxd -i {tflite_model_path} > {c_array_path}"
    print(f"\nRunning command: {command}")
    exit_code_xxd = os.system(command)
    if exit_code_xxd == 0:
        with open(c_array_path, 'r') as f:
            content = f.read()
        content = content.replace(f'unsigned char {os.path.basename(tflite_model_path).replace(".", "_")}', f'const unsigned char {c_var_name}[]')
        content = content.replace(f'unsigned int {os.path.basename(tflite_model_path).replace(".", "_")}_len', f'const unsigned int {c_var_name}_len')
        content = f"""
#ifndef ENCODER_MODEL_DATA_H
#define ENCODER_MODEL_DATA_H

// Auto-generated by xxd. Do not edit manually.
// Model expects SCALED input (e.g., [0, 1])

{content}

#endif // ENCODER_MODEL_DATA_H
"""
        with open(c_array_path, 'w') as f:
            f.write(content)
        print(f"Encoder TFLite model converted to C array: {c_array_path}")
        print(f"C variable name: {c_var_name}")
    else:
        print(f"Warning: 'xxd' command failed (exit code {exit_code_xxd}). C array not generated.")
        print("You may need to install 'xxd' or convert the .tflite file manually.")

except FileNotFoundError:
    print("\nWarning: 'xxd' command not found. Install 'xxd'.")
except Exception as e:
    print(f"\nAn error occurred during C array conversion: {e}")


# --- 6. Export Decoder Model to TensorFlow.js ---
# NOTE: The TF.js model will output SCALED data [0, 1]
print("\n--- Exporting Decoder to TensorFlow.js (outputs scaled data [0,1]) ---")
tfjs_model_path = os.path.join(TFJS_EXPORT_DIR, 'decoder_model_tfjs')
keras_decoder_filename = os.path.join(KERAS_TEMP_DIR, 'decoder.keras') # Use .keras extension

exit_code_tfjs = -1 # Initialize exit code
try:
    # Save the decoder model in the recommended .keras format
    decoder.save(keras_decoder_filename)
    print(f"Decoder Keras model saved temporarily to: {keras_decoder_filename}")

    # Run the converter command using tf_keras format
    # Make sure you have tensorflowjs_converter installed: pip install tensorflowjs
    command = f"tensorflowjs_converter --input_format=tf_keras {keras_decoder_filename} {tfjs_model_path}"
    print(f"\nRunning command: {command}")
    exit_code_tfjs = os.system(command)

    if exit_code_tfjs == 0:
        print(f"Decoder model successfully converted to TensorFlow.js format in: {tfjs_model_path}")
        print("Generated files:")
        for filename in os.listdir(tfjs_model_path):
            print(f"- {filename}")
    else:
        print(f"Error: TensorFlow.js conversion failed (exit code {exit_code_tfjs}).")
        print("Ensure 'tensorflowjs_converter' is installed and in your PATH ('pip install tensorflowjs').")

except Exception as e:
    print(f"\nAn error occurred during TensorFlow.js conversion: {e}")
finally:
    # Clean up the temporary Keras model directory
    try:
        if os.path.exists(KERAS_TEMP_DIR):
             shutil.rmtree(KERAS_TEMP_DIR)
             print(f"Cleaned up temporary Keras directory: {KERAS_TEMP_DIR}")
    except OSError as e:
        print(f"Error removing temporary directory {KERAS_TEMP_DIR}: {e}")


print("\n--- Model Training and Export Complete ---")
print(f"Encoder (TFLite): {tflite_model_path}")
print(f"Encoder (C Array): {c_array_path if exit_code_xxd == 0 else 'Not generated'}")
print(f"Decoder (TF.js): {tfjs_model_path if exit_code_tfjs == 0 else 'Not generated'}")
print("\nIMPORTANT:")
print(" - The TFLite Encoder model expects input data scaled to [0, 1].")
print(" - The TensorFlow.js Decoder model will output data scaled to [0, 1]. You need to inverse_transform it on the server.")