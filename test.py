import numpy as np
import tensorflow as tf
import joblib
import os
import matplotlib.pyplot as plt

# --- Paths ---
KERAS_EXPORT_DIR = 'exported_models/keras'
DECODER_MODEL_PATH = os.path.join(KERAS_EXPORT_DIR, 'decoder_model.keras')
SCALER_PATH = os.path.join(KERAS_EXPORT_DIR, 'scaler.joblib')

TFLITE_ENCODER_PATH = 'exported_models/tflite/encoder_model.tflite'

# --- Load Scaler ---
try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded.")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    exit()

# --- Load TFLite Encoder ---
try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_ENCODER_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite encoder loaded.")
except Exception as e:
    print(f"❌ Error loading TFLite encoder: {e}")
    exit()

# --- Load Keras Decoder ---
try:
    decoder = tf.keras.models.load_model(DECODER_MODEL_PATH)
    print("✅ Keras decoder loaded.")
except Exception as e:
    print(f"❌ Error loading Keras decoder: {e}")
    exit()

# --- Test Samples ---
test_temperatures = np.array([0, 25, 50, 75,32,32.8,32.1, 100, 125, 150], dtype=np.float32).reshape(-1, 1)
scaled = scaler.transform(test_temperatures)

# --- Encode Using TFLite Encoder ---
latent_vectors = []
for s in scaled:
    s = s.reshape(1, -1).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], s)
    interpreter.invoke()
    latent = interpreter.get_tensor(output_details[0]['index'])
    latent_vectors.append(latent[0])

latent_vectors = np.array(latent_vectors)

# --- Decode Using Keras Decoder ---
reconstructed_scaled = decoder.predict(latent_vectors)
reconstructed_temp = scaler.inverse_transform(reconstructed_scaled)

# --- Results ---
print("\nTemp (°C) → Latent Vector → Reconstructed Temp (°C)")
print("-" * 70)
for i in range(len(test_temperatures)):
    orig = test_temperatures[i][0]
    latent = np.round(latent_vectors[i], 3)
    recon = np.round(reconstructed_temp[i][0], 2)
    print(f"{orig:9.2f}°C → {latent} → {recon:9.2f}°C")

# --- Optional: Plot ---
plt.figure(figsize=(10, 4))
plt.plot(test_temperatures.flatten(), reconstructed_temp.flatten(), 'o-', label="Reconstructed")
plt.plot(test_temperatures.flatten(), test_temperatures.flatten(), 'r--', label="Ideal")
plt.xlabel("Original Temp (°C)")
plt.ylabel("Reconstructed Temp (°C)")
plt.title("Original vs Reconstructed Temperatures")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
