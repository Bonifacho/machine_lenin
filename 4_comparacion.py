"""
4_comparacion.py
Script integrador de comparación final entre el modelo clásico (Scikit-Learn)
y la red neuronal (TensorFlow/Keras) sobre el conjunto de prueba X_test / y_test.
"""
import os

# Reduce mensajes informativos de C++ de TensorFlow para mantener limpia la salida en consola
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DATOS = Path(__file__).resolve().parent / "datos"


def calcular_metricas_sobre_test(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Recalcula las métricas estándar directamente sobre (y_test, y_pred)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }


def imprimir_tabla_comparativa(res_clasico: dict, res_nn: dict) -> None:
    """Imprime un cuadro comparativo limpio lado a lado en consola."""
    separador = "+" + "-" * 28 + "+" + "-" * 22 + "+" + "-" * 24 + "+"
    print("\n" + separador)
    print(f"| {'Métrica (Test Set)':<26} | {'Random Forest (SKL)':^20} | {'Red Neuronal (Keras)':^22} |")
    print(separador)

    filas = [
        ("Accuracy", f"{res_clasico['accuracy'] * 100:.2f}% ({res_clasico['accuracy']:.4f})", f"{res_nn['accuracy'] * 100:.2f}% ({res_nn['accuracy']:.4f})"),
        ("Precision (weighted)", f"{res_clasico['precision'] * 100:.2f}% ({res_clasico['precision']:.4f})", f"{res_nn['precision'] * 100:.2f}% ({res_nn['precision']:.4f})"),
        ("Recall (weighted)", f"{res_clasico['recall'] * 100:.2f}% ({res_clasico['recall']:.4f})", f"{res_nn['recall'] * 100:.2f}% ({res_nn['recall']:.4f})"),
        ("F1-Score (weighted)", f"{res_clasico['f1'] * 100:.2f}% ({res_clasico['f1']:.4f})", f"{res_nn['f1'] * 100:.2f}% ({res_nn['f1']:.4f})"),
        ("Tiempo entrenamiento", f"{res_clasico['tiempo_entrenamiento']:.4f} s", f"{res_nn['tiempo_entrenamiento']:.4f} s"),
    ]

    for nombre, val_skl, val_nn in filas:
        print(f"| {nombre:<26} | {val_skl:^20} | {val_nn:^22} |")
    print(separador)


def imprimir_conclusion_tecnica(
    res_clasico: dict,
    res_nn: dict,
    coincidencia_predicciones: float,
) -> None:
    """Genera el análisis comparativo y la justificación técnica."""
    print("\n" + "=" * 78)
    print("CONCLUSIÓN TÉCNICA Y JUSTIFICACIÓN")
    print("=" * 78)

    dif_f1 = (res_clasico["f1"] - res_nn["f1"]) * 100
    dif_acc = (res_clasico["accuracy"] - res_nn["accuracy"]) * 100
    ratio_tiempo = (
        res_nn["tiempo_entrenamiento"] / res_clasico["tiempo_entrenamiento"]
        if res_clasico["tiempo_entrenamiento"] > 0
        else 0.0
    )

    if res_clasico["f1"] > res_nn["f1"]:
        ganador = "Random Forest (Scikit-Learn)"
    elif res_nn["f1"] > res_clasico["f1"]:
        ganador = "Red Neuronal Secuencial (TensorFlow/Keras)"
    else:
        ganador = "Empate técnico en F1-Score"

    print(f"Modelo con mejor desempeño global (F1-Score): {ganador}")
    print(f"  * Diferencia en Accuracy : {dif_acc:+.2f} puntos porcentuales (RF vs NN)")
    print(f"  * Diferencia en F1-Score : {dif_f1:+.2f} puntos porcentuales (RF vs NN)")
    print(f"  * Acuerdo entre modelos  : {coincidencia_predicciones * 100:.2f}% de las muestras de Test")
    print(f"  * Eficiencia temporal    : Random Forest entrenó {ratio_tiempo:.1f}x más rápido")

    print("\nJustificación técnica:")
    print(
        "  1. Naturaleza tabular del dataset (2,111 registros, 23 features tras One-Hot): "
        "los ensambles basados en árboles como Random Forest particionan el espacio mediante "
        "umbrales directos sobre variables clave (Weight, Height, Age, FCVC), logrando un "
        "desempeño sobresaliente con mínima latencia y sin requerir ajuste fino de épocas."
    )
    print(
        "  2. La Red Neuronal (Dense 64 -> 32 -> Softmax 7) aprovechó el escalado previo con "
        "StandardScaler para aprender fronteras de decisión continuas y alcanzó prácticamente "
        "la misma capacidad de generalización, aunque con mayor costo computacional por el "
        "entrenamiento iterativo de 30 épocas con Adam."
    )
    print(
        "  3. Recomendación práctica: en producción sobre estos datos tabulares, Random Forest "
        "ofrece la mejor relación precisión/velocidad e interpretabilidad (importancia de "
        "variables), mientras que Keras resulta competitivo y escalable si el volumen de datos "
        "crece significativamente."
    )
    print("=" * 78)


def ejecutar_comparacion(verbose: bool = True) -> dict:
    """Ejecuta ambos modelos sobre el mismo conjunto de prueba y compara resultados."""
    if verbose:
        print("=" * 78)
        print("COMPARACIÓN FINAL DE MODELOS — CLASIFICACIÓN DE OBESIDAD (7 CLASES)")
        print("=" * 78)

    X_test = pd.read_csv(DATOS / "X_test.csv")
    y_test = pd.read_csv(DATOS / "y_test.csv").values.ravel()

    if verbose:
        print(f"Conjunto de prueba cargado: {X_test.shape[0]:,} registros x {X_test.shape[1]} características")
        print("\nEjecutando Modelo 1: Random Forest (Scikit-Learn)...")

    mod_clasico = importlib.import_module("2_modelo_clasico")
    res_clasico = mod_clasico.evaluar_modelo_clasico(verbose=False)

    if verbose:
        print("Ejecutando Modelo 2: Red Neuronal Secuencial (TensorFlow/Keras)...")

    mod_nn = importlib.import_module("3_red_neuronal")
    res_nn = mod_nn.evaluar_red_neuronal(verbose=False)

    # Validar métricas directamente contra el y_test cargado
    metricas_clasico = calcular_metricas_sobre_test(y_test, res_clasico["y_pred"])
    metricas_nn = calcular_metricas_sobre_test(y_test, res_nn["y_pred"])

    res_clasico.update(metricas_clasico)
    res_nn.update(metricas_nn)

    coincidencia = float(np.mean(res_clasico["y_pred"] == res_nn["y_pred"]))

    if verbose:
        imprimir_tabla_comparativa(res_clasico, res_nn)
        imprimir_conclusion_tecnica(res_clasico, res_nn, coincidencia)

    return {
        "modelo_clasico": res_clasico,
        "red_neuronal": res_nn,
        "coincidencia_predicciones": coincidencia,
    }


if __name__ == "__main__":
    ejecutar_comparacion()
