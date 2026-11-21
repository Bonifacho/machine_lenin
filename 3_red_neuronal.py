"""
3_red_neuronal.py
Red neuronal (TensorFlow/Keras) para clasificar el nivel de obesidad (7 clases).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carpeta datos/ (funciona sin importar desde dónde se ejecute)
DATOS = Path(__file__).resolve().parent / "datos"


def evaluar_red_neuronal(verbose=True):
    # Semilla fija (dentro de la función para que dé siempre el mismo resultado,
    # aunque 4_comparacion.py ejecute otros modelos antes)
    tf.keras.utils.set_random_seed(412)  # fija la semilla de Python, NumPy y TensorFlow

    # 1. Cargar los datos ya preprocesados
    X_train = pd.read_csv(DATOS / "X_train.csv").values.astype("float32")
    X_test = pd.read_csv(DATOS / "X_test.csv").values.astype("float32")
    y_train = pd.read_csv(DATOS / "y_train.csv").values.ravel()
    y_test = pd.read_csv(DATOS / "y_test.csv").values.ravel()

    # 2. Definir la red: 2 capas ocultas (ReLU) y salida Softmax con 7 clases
    modelo = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(7, activation="softmax"),
    ])

    # 3. Compilar: Adam + loss para etiquetas enteras 0..6
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])

    # 4. Entrenar (20 % de train se usa como validación)
    if verbose:
        print("Entrenando el modelo...")
    inicio = time.time()
    modelo.fit(X_train, y_train, epochs=30, validation_split=0.2,
               verbose=2 if verbose else 0)
    tiempo = time.time() - inicio
    if verbose:
        print("¡Modelo entrenado!")

    # 5. Evaluar en test
    loss, acc = modelo.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(modelo.predict(X_test, verbose=0), axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    if verbose:
        print(f"Loss en test     : {loss:.4f}")
        print(f"Accuracy         : {accuracy:.4f}")
        print(f"Precision        : {precision:.4f}")
        print(f"Recall           : {recall:.4f}")
        print(f"F1-Score         : {f1:.4f}")

    # 6. Devolver resultados (los usa 4_comparacion.py)
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "y_pred": y_pred, "tiempo_entrenamiento": tiempo}


if __name__ == "__main__":
    evaluar_red_neuronal()
