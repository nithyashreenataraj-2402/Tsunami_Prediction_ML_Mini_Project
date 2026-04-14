from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__)
CORS(app)

model_path = Path(__file__).with_name("tsunami_model.pkl")
model = joblib.load(model_path)


@app.route("/")
def home():
    return jsonify({"message": "Tsunami prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data received"}), 400

        magnitude = float(data["magnitude"])
        depth = float(data["depth"])
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        nst = float(data["nst"])
        dmin = float(data["dmin"])
        gap = float(data["gap"])
        mag_type = str(data["magType"])
        month = int(data["month"])

        depth_magnitude_ratio = depth / (magnitude + 1e-5)
        shallow_quake = int(depth < 70)
        strong_quake = int(magnitude >= 6.5)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        input_df = pd.DataFrame([{
            "magnitude": magnitude,
            "depth": depth,
            "latitude": latitude,
            "longitude": longitude,
            "nst": nst,
            "dmin": dmin,
            "gap": gap,
            "magType": mag_type,
            "month": month,
            "depth_magnitude_ratio": depth_magnitude_ratio,
            "shallow_quake": shallow_quake,
            "strong_quake": strong_quake,
            "month_sin": month_sin,
            "month_cos": month_cos
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": round(float(probability) * 100, 2),
            "label": "Tsunami" if prediction == 1 else "No Tsunami"
        })

    except KeyError as e:
        return jsonify({"error": f"Missing input field: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid input value: {str(e)}"}), 400
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
