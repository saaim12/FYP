import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import json
# Import MinMaxScaler for data scaling
from sklearn.preprocessing import MinMaxScaler
import joblib # For saving the scaler
import shutil # For cleaning up directories
import sys # To get python executable path
import subprocess # For running xxd reliably

# --- Configuration ---
NUM_SAMPLES = 5000       # Number of training samples
EPOCHS = 200             # <<<<<< INCREASED Epochs for potentially better convergence
BATCH_SIZE = 32          # Training batch size
TEMP_MIN = 0.0           # Minimum temperature (°C)
TEMP_MAX = 50.0          # Maximum temperature (°C)
LATENT_DIM = 2           # Dimension of the latent space (encoded representation)
NOISE_FACTOR = 0.5       # Add some noise to simulate real-world fluctuations

# --- Output Directories ---
EXPORT_DIR = 'exported_models'
TFLITE_EXPORT_DIR = os.path.join(EXPORT_DIR, 'tflite')
KERAS_EXPORT_DIR = os.path.join(EXPORT_DIR, 'keras') # Directory for Keras models

os.makedirs(TFLITE_EXPORT_DIR, exist_ok=True)
os.makedirs(KERAS_EXPORT_DIR, exist_ok=True) # Create Keras export dir

print(f"TensorFlow Version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# --- 1. Generate Synthetic Temperature Data ---
time = np.linspace(0, 100, NUM_SAMPLES)
base_temp = TEMP_MIN + (TEMP_MAX - TEMP_MIN) / 2 * (1 + np.sin(time * 0.1))
noise = np.random.normal(0, NOISE_FACTOR, NUM_SAMPLES)
temperature_data = base_temp + noise
temperature_data = np.clip(temperature_data, TEMP_MIN, TEMP_MAX)
temperature_data = temperature_data.astype(np.float32).reshape(-1, 1)

print(f"Generated {NUM_SAMPLES} temperature samples.")
print(f"Original data shape: {temperature_data.shape}")
print(f"Example original data points: {temperature_data[:5].flatten()}")

# --- Scale Data to [0, 1] ---
scaler = MinMaxScaler()
scaled_temperature_data = scaler.fit_transform(temperature_data)
# --- Save the scaler ---
scaler_filename = os.path.join(KERAS_EXPORT_DIR, "scaler.joblib")
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to: {scaler_filename}")


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
# plt.show()

# --- 2. Define Autoencoder Model Architecture ---
input_dim = scaled_temperature_data.shape[1]

# --- Encoder ---
# <<<<<< INCREASED NEURONS >>>>>>>
encoder_input = keras.Input(shape=(input_dim,), name='encoder_input')
x = layers.Dense(32, activation='relu')(encoder_input) # Increased neurons
x = layers.Dense(16, activation='relu')(x)             # Increased neurons
encoder_output = layers.Dense(LATENT_DIM, activation='linear', name='encoder_output')(x)
encoder = keras.Model(encoder_input, encoder_output, name='encoder')
print("\n--- Encoder Summary ---")
encoder.summary()

# --- Decoder ---
# <<<<<< INCREASED NEURONS >>>>>>>
decoder_input = keras.Input(shape=(LATENT_DIM,), name='decoder_input')
x = layers.Dense(16, activation='relu')(decoder_input)   # Increased neurons
x = layers.Dense(32, activation='relu')(x)            # Increased neurons
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
autoencoder.compile(optimizer='adam', loss='mse')

print("\n--- Training Autoencoder on SCALED data ---")
history = autoencoder.fit(
    scaled_temperature_data,
    scaled_temperature_data,
    epochs=EPOCHS, # Using increased epochs
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.2,
    verbose=1
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
# plt.show()

# --- 4. Test Reconstruction ---
print("\n--- Reconstruction Test (Scaled vs Original) ---")
num_test_samples = 10
test_indices = np.random.choice(len(scaled_temperature_data), num_test_samples, replace=False)

original_samples_unscaled = temperature_data[test_indices]
original_samples_scaled = scaled_temperature_data[test_indices]
reconstructed_samples_scaled = autoencoder.predict(original_samples_scaled)
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
print("\n--- Exporting Encoder to TFLite (expects scaled input [0,1]) ---")
converter = tf.lite.TFLiteConverter.from_keras_model(encoder)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

tflite_model_path = os.path.join(TFLITE_EXPORT_DIR, 'encoder_model.tflite')
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
print(f"Encoder TFLite model saved to: {tflite_model_path}")

# --- Convert TFLite model to C array ---
c_array_path = os.path.join(TFLITE_EXPORT_DIR, 'encoder_model_data.cc')
c_var_name = 'g_encoder_model_data'
exit_code_xxd = -1
try:
    # Use 'where' on Windows or 'which' on Unix-like systems to find xxd
    find_cmd = 'where' if sys.platform == 'win32' else 'which'
    # Use subprocess.run to find the command path reliably
    result = subprocess.run([find_cmd, 'xxd'], capture_output=True, text=True, check=False, shell=True) # Use shell=True for where/which
    # Handle potential multiple lines or errors in finding xxd
    xxd_path = ""
    if result.returncode == 0 and result.stdout:
        xxd_path = result.stdout.strip().splitlines()[0] # Take the first line

    if not xxd_path:
        print("\nWarning: 'xxd' command not found in PATH.")
        raise FileNotFoundError

    print(f"Found xxd at: {xxd_path}")

    # Prepare command as a list for subprocess.run
    command_list = [xxd_path, "-i", tflite_model_path]
    print(f"\nRunning command: {' '.join(command_list)} > {c_array_path}")

    with open(c_array_path, 'w') as f_out:
        # Run xxd and redirect stdout to the file
        process = subprocess.run(command_list, stdout=f_out, text=True, check=True) # check=True raises exception on error

    exit_code_xxd = process.returncode # Will be 0 if successful

    # Post-process the generated C array file (if successful)
    with open(c_array_path, 'r') as f:
        content = f.read()
    # Sanitize variable name from path
    base_tflite_name = os.path.basename(tflite_model_path).replace(".", "_").replace("-", "_")
    content = content.replace(f'unsigned char {base_tflite_name}', f'const unsigned char {c_var_name}[]')
    content = content.replace(f'unsigned int {base_tflite_name}_len', f'const unsigned int {c_var_name}_len')
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

except FileNotFoundError:
     print("Ensure 'xxd' is installed and in your PATH (e.g., via Git for Windows). C array not generated.")
except subprocess.CalledProcessError as e:
     print(f"Error: 'xxd' command failed (exit code {e.returncode}). C array not generated.")
     print(f"Stderr: {e.stderr}")
except Exception as e:
    print(f"\nAn error occurred during C array conversion: {e}")


# --- 6. Export Decoder Model to Keras format ---
print("\n--- Exporting Decoder to Keras format (outputs scaled data [0,1]) ---")
keras_decoder_path = os.path.join(KERAS_EXPORT_DIR, 'decoder_model.keras')
try:
    decoder.save(keras_decoder_path)
    print(f"Decoder Keras model saved successfully to: {keras_decoder_path}")
except Exception as e:
    print(f"\nAn error occurred during Keras decoder saving: {e}")
    keras_decoder_path = None # Indicate saving failed


print("\n--- Model Training and Export Complete ---")
print(f"Scaler: {scaler_filename}")
print(f"Encoder (TFLite): {tflite_model_path}")
print(f"Encoder (C Array): {c_array_path if exit_code_xxd == 0 else 'Not generated'}")
print(f"Decoder (Keras): {keras_decoder_path if keras_decoder_path else 'Not generated'}")
print("\nIMPORTANT:")
print(" - The TFLite Encoder model expects input data scaled to [0, 1].")
print(" - The Keras Decoder model will output data scaled to [0, 1]. You need to load the scaler and inverse_transform it on the server.")

# Add plt.show() at the end if you want to see the plots after everything runs
plt.show()
