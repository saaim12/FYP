# Secure Sensor Data Transmission using Autoencoder on ESP32

This project demonstrates a system for securely transmitting temperature data from a DHT11 sensor connected to an ESP32 microcontroller to a Python Flask server. The core security mechanism relies on an autoencoder neural network: the temperature reading is encoded into a lower-dimensional latent vector on the ESP32 before transmission. The server then uses a corresponding decoder model to reconstruct the original temperature.

This method provides a level of data obfuscation, as the transmitted latent vector is meaningless without the specific decoder model.

---

## 📚 Concepts

### 1. Autoencoders
- A type of unsupervised neural network used for efficient data codings (dimensionality reduction).
- Consists of:
  - **Encoder**: Compresses the input data into a lower-dimensional latent representation.
  - **Decoder**: Reconstructs the original data from the latent representation.
- In this project:
  - The **encoder** runs on the ESP32.
  - The **decoder** runs on the Flask server.

### 2. TensorFlow Lite for Microcontrollers (TFLM)
- A lightweight ML framework designed for microcontrollers like the ESP32.
- Enables on-device inference using a trained encoder model.

### 3. Data Obfuscation
- Instead of sending raw temperature, the encoded latent vector is sent.
- Anyone intercepting traffic sees only encoded data, not meaningful temperature.
- **Note**: This is not strong encryption. Use TLS/HTTPS for real-world security.

### 4. Client-Server Architecture
- **ESP32** (Client): Reads sensor data, encodes it, sends it via WiFi.
- **Flask Server**: Receives encoded data, decodes it, and returns the temperature.

---

## 🧠 System Architecture

### 1. Model Training Script (`autoencoder_training_py.py`)
- Trains an autoencoder using synthetic or real data.
- Saves:
  - `encoder_model.tflite` (converted to C array for ESP32)
  - `decoder_model.keras` (for the Flask server)
  - `scaler.joblib` (for scaling and inverse transformation)

### 2. ESP32 Client (`esp32_encoder_client.ino`)
- Loads the encoder model header file.
- Reads data from the DHT11 sensor.
- Scales and encodes the temperature.
- Sends the encoded data as JSON to the Flask server.

### 3. Python Flask Server (`server.py`)
- Loads decoder model and scaler.
- Accepts POST requests with encoded data.
- Decodes and scales back to the original temperature.
- Responds with the reconstructed temperature.

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.8+
- Git (for `xxd`)
- Arduino IDE (with ESP32 support) or PlatformIO
- ESP32 board + DHT11 sensor + USB cable

### 1. Clone the Repository
```bash
git clone https://github.com/saaim12/AutoEncoders-bases-encryption-training-and-fitting-in-devices-such-as-esp-or-rasberry-pi.git
cd <your-directory-name>
