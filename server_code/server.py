import flask
from flask import request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib # To load the scaler
import os
import logging

# --- Configuration ---
MODEL_DIR = '../exported_models/keras' # Directory where decoder and scaler are saved
DECODER_MODEL_PATH = os.path.join(MODEL_DIR, 'decoder_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
HOST = '0.0.0.0' # Listen on all available network interfaces
PORT = 5000      # Port the server will listen on

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Flask App ---
app = flask.Flask(__name__)

# --- Global variables for model and scaler ---
decoder_model = None
scaler = None

# --- Load Model and Scaler ---
def load_resources():
    """Loads the decoder model and scaler from disk."""
    global decoder_model, scaler
    try:
        # Check if files exist
        if not os.path.exists(DECODER_MODEL_PATH):
            logging.error(f"Decoder model file not found at: {DECODER_MODEL_PATH}")
            raise FileNotFoundError(f"Decoder model file not found at: {DECODER_MODEL_PATH}")
        if not os.path.exists(SCALER_PATH):
            logging.error(f"Scaler file not found at: {SCALER_PATH}")
            raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")

        logging.info(f"Loading Keras decoder model from: {DECODER_MODEL_PATH}")
        decoder_model = keras.models.load_model(DECODER_MODEL_PATH)
        logging.info("Decoder model loaded successfully.")
        decoder_model.summary() # Print model summary to confirm

        logging.info(f"Loading scaler from: {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
        logging.info("Scaler loaded successfully.")

    except Exception as e:
        logging.error(f"Error loading resources: {e}", exc_info=True)
        # Exit if resources can't be loaded, as the server can't function
        exit(1)

# --- Define API Endpoint ---
@app.route('/decode', methods=['POST'])
def decode_temperature():
    """
    API endpoint to receive the latent vector, decode it, and return temperature.
    Expects JSON payload like: {"latent_vector": [val1, val2]}
    """
    logging.info(f"Received request on /decode from {request.remote_addr}")

    # Check if model and scaler are loaded
    if decoder_model is None or scaler is None:
        logging.error("Model or scaler not loaded. Cannot process request.")
        return jsonify({"error": "Server error: Model or scaler not loaded"}), 500

    # Get JSON data from the request
    data = request.get_json()

    if not data:
        logging.warning("Request received without JSON data.")
        return jsonify({"error": "Missing JSON payload"}), 400

    # Extract latent vector
    if 'latent_vector' not in data:
        logging.warning("Request received without 'latent_vector' key in JSON.")
        return jsonify({"error": "Missing 'latent_vector' in JSON payload"}), 400

    latent_vector = data['latent_vector']

    # --- Input Validation ---
    # Check if it's a list or tuple
    if not isinstance(latent_vector, (list, tuple)):
         logging.warning(f"Invalid latent_vector type: {type(latent_vector)}. Expected list or tuple.")
         return jsonify({"error": "Invalid latent_vector format: Expected a list/array."}), 400

    # Check if the length matches the decoder's expected input dimension
    try:
        # Get expected input shape (e.g., (None, 2))
        expected_dim = decoder_model.input_shape[1]
        if len(latent_vector) != expected_dim:
            logging.warning(f"Incorrect latent_vector dimension. Received {len(latent_vector)}, expected {expected_dim}.")
            return jsonify({"error": f"Invalid latent_vector dimension. Expected {expected_dim} elements."}), 400
    except Exception as e:
         logging.error(f"Could not determine model input shape: {e}")
         return jsonify({"error": "Server error: Could not validate input shape"}), 500


    # Convert latent vector to numpy array for the model
    try:
        # Ensure data is float32, matching training
        latent_vector_np = np.array(latent_vector, dtype=np.float32).reshape(1, -1)
        logging.info(f"Received latent vector: {latent_vector_np.tolist()}") # Log the received vector
    except ValueError as e:
        logging.warning(f"Could not convert latent_vector to numpy array: {e}")
        return jsonify({"error": "Invalid data in latent_vector. Ensure all elements are numbers."}), 400

    # --- Decode ---
    try:
        # Predict the scaled temperature
        scaled_prediction = decoder_model.predict(latent_vector_np)
        logging.info(f"Model predicted scaled value: {scaled_prediction[0][0]}")

        # Inverse transform to get the original temperature scale
        reconstructed_temp = scaler.inverse_transform(scaled_prediction)
        # Extract the single temperature value
        final_temp = reconstructed_temp[0][0]
        logging.info(f"temperature Received")
        print("Reconstructed temp == " + str(round(float(final_temp), 2)))
        # --- Return Response ---
        return jsonify({"message": "Decoded successfully.",'reconstructed_temperature_celsius':str(round(float(final_temp), 2))})


    except Exception as e:
        logging.error(f"Error during decoding or inverse transform: {e}", exc_info=True)
        return jsonify({"error": "Server error during processing"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Load the model and scaler when the script starts
    load_resources()

    # Start the Flask development server
    logging.info(f"Starting Flask server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False) # Set debug=False for production
