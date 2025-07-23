import flask
from flask import request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib  # To load the scaler
import os
import logging


# --- Terminal Text Styling ---
class TerminalColors:
    HEADER = '\033[95m'  # Light magenta
    OKBLUE = '\033[94m'  # Blue
    OKCYAN = '\033[96m'  # Cyan
    OKGREEN = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'  # Red
    ENDC = '\033[0m'  # Reset to normal
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# --- Configuration ---
MODEL_DIR = '../exported_models/keras'
DECODER_MODEL_PATH = os.path.join(MODEL_DIR, 'decoder_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
HOST = '0.0.0.0'
PORT = 5000

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Flask App ---
app = flask.Flask(__name__)

# --- Global variables ---
decoder_model = None
scaler = None


# --- Load Model and Scaler ---
def load_resources():
    global decoder_model, scaler
    try:
        if not os.path.exists(DECODER_MODEL_PATH):
            logging.error(f"Decoder model file not found at: {DECODER_MODEL_PATH}")
            raise FileNotFoundError(f"Decoder model file not found at: {DECODER_MODEL_PATH}")
        if not os.path.exists(SCALER_PATH):
            logging.error(f"Scaler file not found at: {SCALER_PATH}")
            raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")

        print(
            f"{TerminalColors.OKBLUE}[INFO]{TerminalColors.ENDC} Loading decoder model from: {TerminalColors.UNDERLINE}{DECODER_MODEL_PATH}{TerminalColors.ENDC}")
        decoder_model = keras.models.load_model(DECODER_MODEL_PATH)
        logging.info("Decoder model loaded successfully.")
        decoder_model.summary()

        print(
            f"{TerminalColors.OKBLUE}[INFO]{TerminalColors.ENDC} Loading scaler from: {TerminalColors.UNDERLINE}{SCALER_PATH}{TerminalColors.ENDC}")
        scaler = joblib.load(SCALER_PATH)
        logging.info("Scaler loaded successfully.")

    except Exception as e:
        print(f"{TerminalColors.FAIL}[ERROR]{TerminalColors.ENDC} Failed to load resources: {e}")
        exit(1)


# --- API Endpoint ---
@app.route('/decode', methods=['POST'])
def decode_temperature():
    logging.info(f"Received request on /decode from {request.remote_addr}")

    if decoder_model is None or scaler is None:
        logging.error("Model or scaler not loaded.")
        return jsonify({"error": "Server error: Model or scaler not loaded"}), 500

    data = request.get_json()
    if not data:
        logging.warning("Missing JSON payload.")
        return jsonify({"error": "Missing JSON payload"}), 400

    if 'latent_vector' not in data:
        logging.warning("Missing 'latent_vector' key.")
        return jsonify({"error": "Missing 'latent_vector' in JSON payload"}), 400

    latent_vector = data['latent_vector']

    if not isinstance(latent_vector, (list, tuple)):
        logging.warning(f"Invalid latent_vector type: {type(latent_vector)}")
        return jsonify({"error": "Invalid latent_vector format: Expected a list/array."}), 400

    try:
        expected_dim = decoder_model.input_shape[1]
        if len(latent_vector) != expected_dim:
            logging.warning(f"Incorrect latent_vector dimension: {len(latent_vector)} (expected {expected_dim})")
            return jsonify({"error": f"Invalid latent_vector dimension. Expected {expected_dim} elements."}), 400
    except Exception as e:
        logging.error(f"Could not determine model input shape: {e}")
        return jsonify({"error": "Server error: Could not validate input shape"}), 500

    try:
        latent_vector_np = np.array(latent_vector, dtype=np.float32).reshape(1, -1)
        print(
            f"{TerminalColors.OKCYAN}[VECTOR]{TerminalColors.ENDC} Received latent vector: {latent_vector_np.tolist()}")
    except ValueError as e:
        logging.warning(f"Could not convert latent_vector to numpy array: {e}")
        return jsonify({"error": "Invalid data in latent_vector. Ensure all elements are numbers."}), 400

    try:
        scaled_prediction = decoder_model.predict(latent_vector_np)
        logging.info(f"Model predicted scaled value: {scaled_prediction[0][0]}")

        reconstructed_temp = scaler.inverse_transform(scaled_prediction)
        final_temp = reconstructed_temp[0][0]

        # --- Beautiful, Colored Output ---
        print(f"\n{TerminalColors.HEADER}{TerminalColors.BOLD}==============================")
        print(f"{TerminalColors.OKGREEN}🎉 SUCCESS: Temperature decoded successfully!")
        print(f"🌡️  Final Temperature: {round(float(final_temp), 2)} °C")
        print(f"{TerminalColors.HEADER}=============================={TerminalColors.ENDC}\n")

        return jsonify({"message": "Decoded successfully, and reconstructed"})

    except Exception as e:
        logging.error(f"Error during decoding or inverse transform: {e}", exc_info=True)
        return jsonify({"error": "Server error during processing"}), 500


# --- Main ---
if __name__ == '__main__':
    load_resources()
    print(f"{TerminalColors.OKBLUE}🚀 Starting server on {HOST}:{PORT}...{TerminalColors.ENDC}")
    app.run(host=HOST, port=PORT, debug=False)
