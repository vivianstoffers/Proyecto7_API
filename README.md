Proyecto 7
Autora: Vivian Stoffers
Bootcamp Ciencia de Datos e Inteligencia Artificial
Universidad del Desarrollo


Descripción del proyecto
Este proyecto implementa una API REST basada en Flask que clasifica reseñas de texto de la Google Play Store según su sentimiento: negativo, neutral o positivo.
El modelo fue entrenado en Python utilizando técnicas de procesamiento de lenguaje natural (NLP) y vectorización TF-IDF, logrando un desempeño sólido en la detección de opiniones.


Objetivo
Desarrollar un modelo de aprendizaje automático capaz de predecir el sentimiento de una reseña y desplegarlo en la nube mediante una API pública accesible para cualquier usuario.


Tecnologías utilizadas
- Python 3
- Flask
- scikit-learn
- joblib
- numpy
- Render (para el despliegue)


Estructura del proyecto
- Proyecto7_API
-- app.py
-- requirements.txt
-- README.md
-- models
---- sentiment_pipeline.joblib
---- classes.joblib
-- utils


Modelo
El modelo utiliza una combinación de TF-IDF y Regresión Logística para clasificar las reseñas en tres categorías:

Clase | Descripción
0 | Negativo
1 | Neutral
2 | Positivo

Durante las pruebas, la Regresión Logística obtuvo un 77 % de precisión (accuracy) y un F1-score de 0.77, superando al modelo comparativo (SVM lineal).


Uso de la API

URL base:
https://proyecto7-api.onrender.com

1 - Verificar funcionamiento
GET /

Respuesta de ejemplo:
{
"status": "online",
"message": "API de análisis de sentimientos lista para recibir texto."
}

2 - Obtener predicción de sentimiento
POST /predict

Body (JSON):
{"text": "Me encanta esta app, funciona perfecto y es rápida."}

Respuesta de ejemplo:
{
"prediction": "Positive",
"confidence": 0.97,
"probas": {
"Negative": 0.01,
"Neutral": 0.02,
"Positive": 0.97
}
}


Despliegue
El proyecto fue desplegado en Render (plan gratuito).

Comandos de configuración:
Build Command: pip install -r requirements.txt
Start Command: python app.py

Desarrollo en Google Colab
Puedes revisar el análisis exploratorio, la limpieza del texto y el entrenamiento del modelo en el siguiente enlace:
https://colab.research.google.com/drive/1CNcd7Xv8mQPrnSXVBA_BJiu4V3v6XO1c?usp=sharing

La presentación se puede ver en el siguiente enlace
https://docs.google.com/presentation/d/1e-x3mBavvPcAYT6F37f03--6MhnqWQKDuKTvxenBNTE/edit?usp=sharing

