#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h> // For creating JSON payload
#include <DHT.h>         // For DHT sensor

// TensorFlow Lite Micro specific headers
#include <TensorFlowLite_ESP32.h> // Main TFLM header for ESP32
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" // Updated resolver
#include "tensorflow/lite/schema/schema_generated.h"

// Include the model data (ensure this file is in the sketch folder)
#include "encoder_model_data.h" // Contains g_encoder_model_data and g_encoder_model_data_len

//----------------------------------------------------------------
// Configuration Section - MODIFY THESE!
//----------------------------------------------------------------
const char* ssid = "your wifi name";         // Your WiFi network SSID
const char* password = "your wifi password"; // Your WiFi network password

// Server details (use IP address of the machine running the Python server)
// Find your PC's IP address (e.g., using 'ipconfig' on Windows or 'ifconfig'/'ip a' on Linux/macOS)
const char* server_ip = "what ever local ip server shows on running";
const int server_port = 5000; // Port your Flask server is running on
const char* server_endpoint = "/decode"; // Endpoint defined in Flask server

// DHT Sensor Configuration
#define DHTPIN 4       // GPIO pin the DHT11 data pin is connected to (Change if needed)
#define DHTTYPE DHT11  // Define sensor type as DHT11

// Model Configuration (Should match Python training)
const float TEMP_MIN = 0.0;
const float TEMP_MAX = 50.0;
const int LATENT_DIM = 2; // Must match the model's output dimension

// How often to send data (in milliseconds)
const unsigned long sendInterval = 10000; // Send data every 10 seconds
unsigned long lastSendTime = 0;
//----------------------------------------------------------------

// Global variables for TFLM
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// TFLM requires a Tensor Arena: a dedicated memory buffer.
// Size needs careful calculation based on model needs. Start with a reasonable guess.
// Use the TFLM memory planning tools or examples for better estimation if needed.
constexpr int kTensorArenaSize = 4 * 1024; // 4KB arena size (adjust if needed)
uint8_t tensor_arena[kTensorArenaSize];

// Global DHT sensor object
DHT dht(DHTPIN, DHTTYPE);

// WiFi Client
WiFiClient client;
HTTPClient http;

// Helper function to scale temperature
float scaleTemperature(float tempC) {
  // Clip temperature to the expected range first
  tempC = constrain(tempC, TEMP_MIN, TEMP_MAX);
  // Scale to [0, 1]
  return (tempC - TEMP_MIN) / (TEMP_MAX - TEMP_MIN);
}

// Function to setup TFLM
void setupTfLite() {
  // Set up logging (use standard TF error reporter)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  error_reporter->Report("TFLM Setup Started");

  // Map the model into a usable data structure
  model = tflite::GetModel(exported_models_tflite_encoder_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal "
                           "to supported version %d.",
                           model->version(), TFLITE_SCHEMA_VERSION);
    return; // Indicate failure
  }
   error_reporter->Report("Model mapped successfully.");

  // Use MicroMutableOpResolver. It provides more flexibility by allowing
  // you to register only the operations your model needs.
  // Add ALL the ops used by your encoder model.
  // Common ops for simple Dense models: Relu, FullyConnected, Reshape (sometimes implicit)
  // Check your model structure if unsure (e.g., using Netron app on the .tflite file)
  static tflite::MicroMutableOpResolver<4> micro_op_resolver; // Adjust size (4) if more ops needed
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddFullyConnected();
  // Add other ops if your model uses them (e.g., micro_op_resolver.AddReshape();)

  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return; // Indicate failure
  }
  error_reporter->Report("Tensors allocated successfully.");

  // Get pointers to the model's input and output tensors
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  // Check tensor shapes and types (optional but good for debugging)
  error_reporter->Report("Input tensor: dims=%d, shape=[%d], type=%d",
                         model_input->dims->size, model_input->dims->data[1], model_input->type); // Assuming 2D input (Batch, Features)
  error_reporter->Report("Output tensor: dims=%d, shape=[%d], type=%d",
                         model_output->dims->size, model_output->dims->data[1], model_output->type); // Assuming 2D output (Batch, LatentDim)

   // Check if output dimension matches LATENT_DIM
   if (model_output->dims->data[1] != LATENT_DIM) {
      error_reporter->Report("Error: Model output dimension (%d) does not match configured LATENT_DIM (%d)",
                             model_output->dims->data[1], LATENT_DIM);
      while(1);
   }

   error_reporter->Report("TFLM Setup Complete");
}


// Function to connect to WiFi
void setupWifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  int retries = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    retries++;
    if (retries > 20) { // Timeout after ~10 seconds
        Serial.println("\nFailed to connect to WiFi. Please check credentials or network.");
        // Optional: Enter a deep sleep or halt here
        ESP.restart(); // Restart ESP if connection fails
    }
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

// Function to send data to server
void sendDataToServer(float latent_vec[]) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected. Cannot send data.");
    // Optional: try reconnecting
    // setupWifi();
    return;
  }

  // Construct the server URL
  String serverUrl = "http://" + String(server_ip) + ":" + String(server_port) + String(server_endpoint);
  Serial.print("Sending data to: ");
  Serial.println(serverUrl);

  // Create JSON document
  // Calculate required size: https://arduinojson.org/v6/assistant/
  // Example: {"latent_vector": [1.23, 4.56]} -> approx 50-60 bytes
  const size_t capacity = JSON_ARRAY_SIZE(LATENT_DIM) + JSON_OBJECT_SIZE(1) + 60; // Add some buffer
  DynamicJsonDocument jsonDoc(capacity);

  // Create the JSON array for the latent vector
  JsonArray latentArray = jsonDoc.createNestedArray("latent_vector");
  for (int i = 0; i < LATENT_DIM; ++i) {
    // Add float values directly
    latentArray.add(latent_vec[i]);
  }

  // Serialize JSON to string
  String jsonPayload;
  serializeJson(jsonDoc, jsonPayload);
  Serial.print("JSON Payload: ");
  Serial.println(jsonPayload);


  // Send HTTP POST request
  http.begin(client, serverUrl); // Use WiFiClient for stability
  http.addHeader("Content-Type", "application/json");

  int httpResponseCode = http.POST(jsonPayload);

  Serial.print("HTTP Response code: ");
  Serial.println(httpResponseCode);

  if (httpResponseCode > 0) {
    String responsePayload = http.getString();
    Serial.print("Response payload: ");
    Serial.println(responsePayload);
    // You could potentially parse this response if needed
    // Example: Deserialize the JSON response
    // DynamicJsonDocument responseDoc(128); // Adjust size as needed
    // DeserializationError error = deserializeJson(responseDoc, responsePayload);
    // if (!error) {
    //   float temp = responseDoc["reconstructed_temperature_celsius"];
    //   Serial.print("Server decoded temp: "); Serial.println(temp);
    // } else {
    //   Serial.print("deserializeJson() failed: "); Serial.println(error.c_str());
    // }

  } else {
    Serial.printf("HTTP POST failed, error: %s\n", http.errorToString(httpResponseCode).c_str());
  }

  http.end(); // Free resources
}


//----------------------------------------------------------------
// Arduino Setup Function
//----------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("ESP32 Autoencoder Client Starting...");

  // Initialize DHT sensor
  dht.begin();
  Serial.println("DHT11 Initialized.");

  // Connect to WiFi
  setupWifi();

  // Setup TensorFlow Lite Micro
  setupTfLite();

  // Check if TFLM setup was successful
  if (interpreter == nullptr || model_input == nullptr || model_output == nullptr) {
      Serial.println("!!! TFLM Setup Failed - Halting !!!");
      while(1); // Halt execution
  }

  Serial.println("Setup complete. Starting main loop.");
}

//----------------------------------------------------------------
// Arduino Loop Function
//----------------------------------------------------------------
void loop() {
  unsigned long currentTime = millis();

  // Check if it's time to send data
  if (currentTime - lastSendTime >= sendInterval) {
    lastSendTime = currentTime;

    // Read temperature from DHT11
    // Reading temperature or humidity takes about 250 milliseconds!
    // Sensor readings may also be up to 2 seconds 'old' (its a very slow sensor)
    float tempC = dht.readTemperature(); // Read temperature as Celsius (the default)

    // Check if any reads failed and exit early (to try again).
    if (isnan(tempC)) {
      Serial.println("Failed to read from DHT sensor!");
      return; // Skip this loop iteration
    }

    Serial.print("Temperature Read: ");
    Serial.print(tempC);
    Serial.println(" °C");

    // Scale the temperature for the model (to range [0, 1])
    float scaledTemp = scaleTemperature(tempC);
    Serial.print("Scaled Temperature (0-1): ");
    Serial.println(scaledTemp);

    // --- Run TFLM Inference ---
    // Place the scaled temperature into the model's input tensor
    // Ensure input type matches (should be float32 based on training)
    if (model_input->type == kTfLiteFloat32) {
         model_input->data.f[0] = scaledTemp; // Assuming input shape is [1, 1]
         Serial.println("Input tensor populated.");
    } else {
        Serial.println("Error: Model input tensor is not Float32!");
        return; // Skip inference if type mismatch
    }


    // Run the model interpreter
    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on input: %f\n", scaledTemp);
      Serial.println("!!! Model Invoke Failed !!!");
      return; // Skip sending data if inference fails
    }
    Serial.println("Model invoked successfully.");

    // Extract the output (latent vector) from the output tensor
    // Ensure output type matches (should be float32 based on training)
    if (model_output->type == kTfLiteFloat32) {
        float latent_vector[LATENT_DIM];
        for (int i = 0; i < LATENT_DIM; ++i) {
             latent_vector[i] = model_output->data.f[i]; // Assuming output shape is [1, LATENT_DIM]
        }

        Serial.print("Encoded Latent Vector: [");
        for (int i = 0; i < LATENT_DIM; ++i) {
            Serial.print(latent_vector[i], 4); // Print with 4 decimal places
            if (i < LATENT_DIM - 1) Serial.print(", ");
        }
        Serial.println("]");

        // Send the latent vector to the server
        sendDataToServer(latent_vector);

    } else {
         Serial.println("Error: Model output tensor is not Float32!");
         return; // Skip sending if type mismatch
    }

    Serial.println("----------------------------------------");

  } // End of interval check

  // Add a small delay to prevent busy-waiting and allow other tasks
  delay(10);
}
