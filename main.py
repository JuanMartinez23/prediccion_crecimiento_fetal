from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import uuid


# ✅ Definir la clase usada para crear el modelo FCM
class MapaCognitivoDifuso:
    def predict(self, X):
        X = np.array(X)
        return np.where(X[:, 1] > 30, 1, 0)  # ejemplo simple: si IMC > 30 = FGR

app = Flask(__name__)

# Cargar modelos previamente entrenados
modelos = {
    "Regresión logística": joblib.load("models/logistic_model.pkl"),
    "Red neuronal artificial": joblib.load("models/mlp_model.pkl"),
    "Máquina de vector soporte": joblib.load("models/svm_model.pkl"),
    "Mapa cognitivo difuso": joblib.load("models/fcm_model.pkl")
}

columnas = [
    "Age", "BMI", "Gestational age of delivery", "Gravidity", "Parity",
    "Initial onset symptoms", "Gestational age of hypertension onset",
    "Gestational age of proteinuria onset", "Past history",
    "Maximum systolic blood pressure", "Maximum diastolic blood pressure",
    "Maximum values of creatinine", "Maximum proteinuria value"
]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/manual', methods=['GET', 'POST'])
def manual_prediction():
    if request.method == 'POST':
        try:
            datos = [
                float(request.form['Age']),
                float(request.form['BMI']),
                float(request.form['Gestational age of delivery']),
                float(request.form['Gravidity']),
                float(request.form['Parity']),
                int(request.form['Initial onset symptoms']),
                float(request.form['Gestational age of hypertension onset']),
                float(request.form['Gestational age of proteinuria onset']),
                int(request.form['Past history']),
                float(request.form['Maximum systolic blood pressure']),
                float(request.form['Maximum diastolic blood pressure']),
                float(request.form['Maximum values of creatinine']),
                float(request.form['Maximum proteinuria value'])
            ]
            modelo_nombre = request.form['modelo']
            modelo = modelos[modelo_nombre]
            prediccion = modelo.predict([datos])[0]
            return render_template("result.html", resultado="Normal" if prediccion == 0 else "FGR", modelo=modelo_nombre)
        except Exception as e:
            return f"Error en los datos: {str(e)}"
    return render_template("manual_prediction.html")

@app.route('/batch', methods=['GET', 'POST'])
def batch_prediction():
    if request.method == 'POST':
        file = request.files['dataset']
        if not file:
            return "No se subió ningún archivo"
        df = pd.read_excel(file)
        modelo_nombre = request.form['modelo']
        modelo = modelos[modelo_nombre]
        try:
            X = df[columnas]
            y = df["Fetal weight"]
            y_true = (y < 2500).astype(int)  # Etiquetar como 1 (FGR) si < 2500g

            y_pred = modelo.predict(X)
            exactitud = (y_true == y_pred).mean()

            # Matriz de confusión
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'FGR'], yticklabels=['Normal', 'FGR'])
            plt.xlabel('Predicción')
            plt.ylabel('Real')
            plt.title('Matriz de Confusión')

            # Guardar imagen con nombre único
            img_id = str(uuid.uuid4()) + ".png"
            img_path = os.path.join('static', 'images', img_id)
            plt.savefig(img_path)
            plt.close()

            return render_template("result.html", resultado=f"Exactitud del modelo: {exactitud:.2%}",
                                   modelo=modelo_nombre, imagen=img_id)
        except Exception as e:
            return f"Error procesando el archivo: {str(e)}"
    return render_template("batch_prediction.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
