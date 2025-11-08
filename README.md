Proyecto 7 – Vivian Stoffers

Este proyecto implementa una API REST basada en Flask que clasifica reseñas de texto de la Google Play Store según su polaridad: negativa, neutral o positiva. El modelo fue entrenado en Python utilizando técnicas de procesamiento de lenguaje natural (NLP) y vectorización TF-IDF.


Objetivo
Desarrollar un modelo de machine learning capaz de predecir el sentimiento de una reseña y desplegarlo en la nube mediante una API pública.


Tecnologías utilizadas
- Python 3
- Flask
- scikit-learn
- joblib
- numpy
- Render (despliegue)


Estructura del proyecto
- Proyecto7_API
-- app.py
-- requirements.txt
-- README.md
-- models/
----sentiment_pipeline.joblib
----classes.joblib
--utils/


Modelo
El modelo utiliza una combinación de TF-IDF y Regresión Logística para clasificar en tres categorías:

0 → Negativo
1 → Neutral
2 → Positivo


Uso de la API
URL base
https://proyecto7-api.onrender.com

1) Verificar funcionamiento

GET /

Respuesta de ejemplo
{
  "message": "API de Sentimiento funcionando correctamente",
  "endpoints": ["/predict"]
}

2) Obtener predicción de sentimiento

POST /predict

Body (JSON)
{
  "text": "This app is amazing, I love using it every day!"
}

Respuesta de ejemplo
{
  "n": 1,
  "results": [
    {
      "texto": "This app is amazing, I love using it every day!",
      "prediccion": "Positivo",
      "confianza": 0.94,
      "probabilidades": {
        "Negativo": 0.02,
        "Neutral": 0.04,
        "Positivo": 0.94
      }
    }
  ]
}


Despliegue
El proyecto fue desplegado en Render (plan gratuito).

Comandos de configuración:
Build Command: pip install -r requirements.txt
Start Command: python app.py


Autora

Vivian Stoffers
Bootcamp Ciencia de Datos e Inteligencia Artificial
Universidad del Desarrollo