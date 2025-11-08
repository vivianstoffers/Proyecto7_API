from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os, traceback

PIPELINE_PATH = "models/sentiment_pipeline.joblib"
CLASSES_PATH  = "models/classes.joblib"

app = Flask(__name__)
CORS(app)

try:
    pipeline = joblib.load(PIPELINE_PATH)
    raw_classes = joblib.load(CLASSES_PATH)
    classes = [c if isinstance(c, str) else str(c) for c in raw_classes]
    print("Modelo y clases cargadas:", classes)
except Exception as e:
    pipeline, classes = None, None
    print("Error al cargar el modelo/clases:", e)

NUMERIC_LABELS = {"0": "Negativo", "1": "Neutral", "2": "Positivo"}
def pretty_label(c):
    c = c if isinstance(c, str) else str(c)
    return NUMERIC_LABELS.get(c, c)

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "API de Sentimiento funcionando correctamente",
        "endpoints": ["/predict"]
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if pipeline is None or classes is None:
            return jsonify({"error": "El modelo no está cargado en el servidor."}), 500

        payload = request.get_json(force=True)
        if "texts" in payload and isinstance(payload["texts"], list):
            texts = [str(t) for t in payload["texts"]]
        elif "text" in payload and isinstance(payload["text"], (str,)):
            texts = [payload["text"]]
        else:
            return jsonify({"error": "Envía 'text' (string) o 'texts' (lista de strings)."}), 400

        raw_pred = pipeline.predict(texts)
        raw_proba = pipeline.predict_proba(texts)

        results = []
        for i, txt in enumerate(texts):
            pred_str = str(raw_pred[i])
            proba_list = [float(x) for x in raw_proba[i]]
            conf = float(max(proba_list))

            probs_pretty = {}
            for j, cls in enumerate(classes):
                probs_pretty[pretty_label(cls)] = float(proba_list[j])

            results.append({
                "texto": txt,
                "prediccion": pretty_label(pred_str),
                "confianza": conf,
                "probabilidades": probs_pretty
            })

        return jsonify({
            "n": len(results),
            "results": results
        }), 200

    except Exception as e:
        print("Error en /predict:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
