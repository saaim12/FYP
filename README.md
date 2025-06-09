# Secure Sensor Data Transmission using Autoencoder on ESP32

This project demonstrates a system for securely transmitting temperature data from a DHT11 sensor connected to an ESP32 microcontroller to a Python Flask server. The core security mechanism relies on an autoencoder neural network: the temperature reading is encoded into a lower-dimensional latent vector on the ESP32 before transmission. The server then uses a corresponding decoder model to reconstruct the original temperature.

This method provides a level of data obfuscation, as the transmitted latent vector is meaningless without the specific decoder model.

## Concepts

1.  **Autoencoders:**
    * A type of unsupervised artificial neural network used for learning efficient data codings (dimensionality reduction).
    * Consists of two parts:
        * **Encoder:** Compresses the input data into a lower-dimensional latent representation (the "code").
        * **Decoder:** Reconstructs the original input data from the latent representation.
    * In this project, the encoder runs on the ESP32 to compress the temperature reading, and the decoder runs on the server to reconstruct it. The compression inherently obfuscates the original value.

2.  **TensorFlow Lite for Microcontrollers (TFLM):**
    * A framework designed to run TensorFlow machine learning models on microcontrollers and other memory-constrained devices like the ESP32.
    * It allows deploying the trained *encoder* model directly onto the ESP32 for on-device inference.

3.  **Data Obfuscation:**
    * The primary security aspect here is obfuscation. The raw temperature value is never transmitted over the network. Instead, the encoded latent vector is sent.
    * An eavesdropper intercepting the network traffic would only see the latent vector values, which are difficult to interpret without access to the trained decoder model.
    * **Note:** This is *not* strong encryption. For robust security in production environments, use standard protocols like HTTPS or MQTT with TLS encryption *in addition* to this obfuscation.

4.  **Client-Server Architecture:**
    * The ESP32 acts as a client, reading sensor data, encoding it, and sending it over WiFi.
    * The Python Flask application acts as a server, listening for incoming requests, receiving the encoded data, decoding it, and making the reconstructed temperature available.

## System Architecture

The system consists of three main components:

1.  **Python Model Training Script (`autoencoder_training_py.py`):**
    * Generates synthetic temperature data (or uses real data if available).
    * Defines and trains the autoencoder model using TensorFlow/Keras.
    * Exports the **encoder** model to TensorFlow Lite format (`.tflite`) and converts it to a C array (`.cc`/`.h`) for the ESP32.
    * Exports the **decoder** model in Keras format (`.keras`) for the Python server.
    * Saves the data scaler (`MinMaxScaler`) used during training, which is needed by the server for inverse transformation.

2.  **ESP32 Client (Arduino C++ - `esp32_encoder_client.ino`):**
    * Includes the C array version of the encoder model (`encoder_model_data.h`).
    * Initializes WiFi connection.
    * Initializes the DHT11 sensor.
    * Initializes TensorFlow Lite Micro with the encoder model.
    * Periodically reads the temperature from the DHT11 sensor.
    * **Scales** the temperature reading to the range [0, 1] (matching the training data).
    * Performs inference using the TFLM encoder to get the latent vector.
    * Formats the latent vector into a JSON payload.
    * Sends the JSON payload via an HTTP POST request to the Python server.

3.  **Python Flask Server (`server.py`):**
    * Loads the saved Keras **decoder** model (`decoder_model.keras`).
    * Loads the saved data **scaler** (`scaler.joblib`).
    * Listens for incoming POST requests on the `/decode` endpoint.
    * Receives the JSON payload containing the `latent_vector` from the ESP32.
    * Uses the decoder model to predict the *scaled* temperature from the latent vector.
    * Uses the loaded scaler to **inverse transform** the scaled prediction back to the original temperature range (°C).
    * Returns the reconstructed temperature in a JSON response.

## Setup Instructions

**Prerequisites:**

* **Python:** Version 3.8 or higher recommended.
* **Git:** Required for the `xxd` command (usually included with Git for Windows) used in the training script.
* **Arduino IDE:** With ESP32 board support installed (use `Tools > Board > Boards Manager...`, search `esp32`, install Espressif Systems package).
* **(Alternative) PlatformIO:** Can be used instead of Arduino IDE (requires setting up `platformio.ini` correctly).
* **ESP32 Board:** Any standard ESP32 development board.
* **DHT11 Sensor:** Connected to the ESP32.
* **USB Cable:** Data-capable USB cable for connecting ESP32.

**Steps:**

1.  **Clone Repository (if applicable):**
    ```bash
    git clone https://github.com/saaim12/AutoEncoders-bases-encryption-training-and-fitting-in-devices-such-as-esp-or-rasberry-pi.git
    cd <what ever name of ypour directory>
    ```

2.  **Setup Python Environment:**
    * Create and activate a virtual environment (recommended):
        ```bash
        python -m venv .venv
        # Windows
        .\.venv\Scripts\activate
        # macOS/Linux
        source .venv/bin/activate
        ```
    * Install required Python libraries:
        ```bash
        pip install tensorflow scikit-learn joblib flask numpy matplotlib
        ```
        *(Optional: Create a `requirements.txt` file with these packages for easier setup)*

3.  **Run Model Training:**
    * Execute the training script:
        ```bash
        python model_training_2.py
        ```
    * This will train the models and create the `exported_models` directory containing:
        * `exported_models/keras/scaler.joblib`
        * `exported_models/keras/decoder_model.keras`
        * `exported_models/tflite/encoder_model.tflite`
        * `exported_models/tflite/encoder_model_data.cc` (which needs renaming)

4.  **Prepare ESP32 Files:**
    * Rename `exported_models/tflite/encoder_model_data.cc` to `encoder_model_data.h`.
    * Create a new sketch in the Arduino IDE (e.g., `esp32_encoder_client`).
    * Copy the code from the `esp32_encoder_client.ino` artifact into your main sketch file.
    * Copy the renamed `encoder_model_data.h` file into the *same folder* as your `.ino` sketch file.

5.  **Setup Arduino IDE:**
    * **Board Selection:** Select `Tools > Board > ESP32 Arduino > ESP32 Dev Module` (or your specific board).
    * **Install Libraries:** Install the following via `Sketch > Include Library > Manage Libraries...`:
        * `DHT sensor library` (by Adafruit)
        * `Adafruit Unified Sensor` (dependency for DHT)
        * `ArduinoJson` (by Benoit Blanchon)
        * `TensorFlowLite_ESP32` (Ensure you have a version compatible with your ESP32 core - you might need to try different versions if you encounter compilation errors like the `flatbuffers` one).
        * (WiFi and HTTPClient are usually included with the ESP32 core).
    * **Restart Arduino IDE** after installing/updating libraries or board packages.

6.  **Configure ESP32 Code (`.ino` file):**
    * Modify the `Configuration Section` at the top:
        * Set `ssid` and `password` for your WiFi network.
        * Set `server_ip` to the **local IP address** of the computer where you will run the Python Flask server. (Find this using `ipconfig` on Windows or `ifconfig`/`ip a` on Linux/macOS).
        * Ensure `DHTPIN` matches the GPIO pin your DHT11 data line is connected to.

## Running the System

1.  **Start the Python Server:**
    * Run the server:
        ```bash
        python server.py
        ```
    * Keep this terminal open. It will log incoming requests and decoding results.

2.  **Compile and Upload to ESP32:**
    * Connect your ESP32 board via USB.
    * In the Arduino IDE, select the correct COM port under `Tools > Port`.
    * Click the "Upload" button. You might need to hold the "BOOT" button during upload if it fails (see previous troubleshooting steps).

3.  **Monitor:**
    * **Server:** Watch the terminal running `server.py` for connection logs and decoded temperature outputs.
    * **ESP32:** Open the Arduino Serial Monitor (`Tools > Serial Monitor`) and set the baud rate to **115200**. Press the ESP32's "RST" button. You should see WiFi connection details, sensor readings, the encoded latent vector, and the response from the server.

## Security Considerations

* The autoencoder provides **obfuscation**, not encryption. A determined attacker with knowledge of the technique *could* potentially attempt to reverse-engineer the model or perform statistical analysis if they capture enough data.
* For sensitive applications, always layer this technique with standard network encryption like **HTTPS** for the HTTP requests or **MQTT with TLS** if using MQTT for communication.

## Future Improvements

* Implement HTTPS on the Flask server and use `WiFiClientSecure` on the ESP32.
* Switch communication from HTTP POST to MQTT for potentially better efficiency and handling of intermittent connections.
* Train the model on actual DHT11 data from your environment instead of synthetic data.
* Experiment with different autoencoder architectures (e.g., convolutional) or latent dimensions.
* Add error handling for more network conditions on the ESP32.
* Implement quantization-aware training for the TFLite model to potentially reduce its size further.
* Add data integrity checks (e.g., checksums or HMAC) to the transmitted payload.
#   F Y P  
 #   F Y P  
 