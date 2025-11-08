from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np
import traceback

PIPELINE_PATH = "Proyecto7_API/models/sentiment_pipeline.joblib"
CLASSES_PATH  = "Proyecto7_API/models/classes.joblib"

try:
    pipeline = joblib.load(PIPELINE_PATH)
    classes  = joblib.load(CLASSES_PATH)
except Exception as e:
    print("Error cargando artefactos:", e)
    pipeline, classes = None, None

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "API de análisis de sentimientos lista para recibir texto."
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if pipeline is None or classes is None:
            return jsonify({"error": "Modelo o clases no cargados"}), 500

        data = request.get_json(force=True)
        text = data.get("text", "")
        if not isinstance(text, str) or text.strip() == "":
            return jsonify({"error": "Debes enviar un texto no vacío en 'text'"}), 400

        probs = pipeline.predict_proba([text])[0]
        pred_idx = int(np.argmax(probs))
        prediction = classes[pred_idx]

        return jsonify({
            "prediction": prediction,
            "confidence": float(probs[pred_idx]),
            "probas": {classes[i]: float(probs[i]) for i in range(len(classes))}
        }), 200

    except Exception as e:
        print("ERROR:", e)
        print(traceback.format_exc())
        return jsonify({"error": "Error interno procesando la solicitud"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
