"""
2_modelo_clasico.py
Modelo clásico de Machine Learning (Random Forest con Scikit-Learn)
para clasificar el nivel de obesidad (7 clases).
"""
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DATOS = Path(__file__).resolve().parent / "datos"

ETIQUETAS_CLASES = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]


def evaluar_modelo_clasico(verbose: bool = True) -> dict:
    """
    Carga los conjuntos preprocesados desde datos/, entrena un
    RandomForestClassifier con semilla fija (random_state=412),
    evalúa sobre Train (control de overfitting) y Test, y retorna
    un diccionario compatible con 4_comparacion.py.
    """
    if verbose:
        print("=" * 72)
        print("1. CARGANDO CONJUNTOS PREPROCESADOS (TRAIN Y TEST)")
        print("=" * 72)

    X_train = pd.read_csv(DATOS / "X_train.csv")
    y_train = pd.read_csv(DATOS / "y_train.csv").values.ravel()
    X_test = pd.read_csv(DATOS / "X_test.csv")
    y_test = pd.read_csv(DATOS / "y_test.csv").values.ravel()

    if verbose:
        print(f"X_train (Entrenamiento): {X_train.shape[0]:,} filas x {X_train.shape[1]} características")
        print(f"y_train (Objetivo)     : {y_train.shape[0]:,} etiquetas (7 clases)")
        print(f"X_test  (Prueba)       : {X_test.shape[0]:,} filas x {X_test.shape[1]} características")
        print(f"y_test  (Objetivo)     : {y_test.shape[0]:,} etiquetas (7 clases)")

        print("\n" + "=" * 72)
        print("2. ENTRENAMIENTO: RANDOM FOREST CLASSIFIER (SCIKIT-LEARN)")
        print("=" * 72)

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=412,
        n_jobs=-1,
    )

    inicio = time.time()
    modelo.fit(X_train, y_train)
    tiempo = time.time() - inicio

    y_pred_train = modelo.predict(X_train)
    y_pred = modelo.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    if verbose:
        print("Modelo RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2, random_state=412) entrenado.")
        print(f"Tiempo de entrenamiento : {tiempo:.4f} s")

        print("\nControl de Overfitting (Train vs Test):")
        print(f"  * Accuracy en Train : {acc_train * 100:.2f}%")
        print(f"  * Accuracy en Test  : {accuracy * 100:.2f}%")
        print(f"  * Diferencia        : {abs(acc_train - accuracy) * 100:.2f}%")

        importancias = pd.Series(modelo.feature_importances_, index=X_train.columns)
        print("\nTop 5 variables más importantes:")
        for var, imp in importancias.sort_values(ascending=False).head(5).items():
            print(f"  * {var:<35}: {imp * 100:.2f}%")

        print("\n" + "=" * 72)
        print("3. EVALUACIÓN EN TEST (X_test)")
        print("=" * 72)
        print(f"Accuracy  (Exactitud)            : {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"Precision (Precisión ponderada)  : {precision:.4f} ({precision * 100:.2f}%)")
        print(f"Recall    (Exhaustividad pond.)  : {recall:.4f} ({recall * 100:.2f}%)")
        print(f"F1-Score  (Ponderado)            : {f1:.4f} ({f1 * 100:.2f}%)")

        print("\nMatriz de Confusión (filas = real, columnas = predicho):")
        print(confusion_matrix(y_test, y_pred))

        print("\nReporte por clase:")
        print(classification_report(y_test, y_pred, target_names=ETIQUETAS_CLASES, digits=4))
        print("=" * 72)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_pred": y_pred,
        "tiempo_entrenamiento": tiempo,
    }


if __name__ == "__main__":
    evaluar_modelo_clasico()
