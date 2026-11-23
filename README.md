# Clasificación del nivel de obesidad: Random Forest vs Red Neuronal

Taller Final Integrador de Machine Learning (601).

**Equipo**

- Integrante A: Andrés Felipe Bonilla Aya
- Integrante B: Omar Bravo Montaña

---

## 1. Pregunta de negocio

> ¿Es posible clasificar con precisión el nivel de obesidad (`NObeyesdad`) de una persona a partir de sus características físicas, hábitos alimenticios y nivel de actividad física, comparando un modelo clásico de Machine Learning frente a una Red Neuronal?

Es un problema de **clasificación multiclase** (7 clases).

## 2. Dataset

Archivo: `datos/datos_obesidad.csv`

Fuente: UCI Machine Learning Repository, dataset 544, "Estimation of obesity levels based on eating habits and physical condition" (Palechor y De la Hoz Manotas, Data in Brief, 2019).

- **2.111 filas** y 17 columnas: **16 variables de entrada** + 1 variable objetivo.
- **0 % de valores nulos**. El docente aprobó trabajar con las 16 variables y sin nulos.
- Variables numéricas (8): `Age`, `Height`, `Weight`, `FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`.
- Variables categóricas (8): `Gender`, `family_history_with_overweight`, `FAVC`, `CAEC`, `SMOKE`, `SCC`, `CALC`, `MTRANS`.
- Variable objetivo `NObeyesdad`, codificada de 0 a 6:

| Código | Clase |
|:-:|---|
| 0 | Insufficient_Weight |
| 1 | Normal_Weight |
| 2 | Overweight_Level_I |
| 3 | Overweight_Level_II |
| 4 | Obesity_Type_I |
| 5 | Obesity_Type_II |
| 6 | Obesity_Type_III |

## 3. Preprocesamiento (`1_preprocesamiento.py`)

1. Carga el CSV original desde `datos/`.
2. Separa `X` (16 variables) e `y` (`NObeyesdad` convertida a números 0..6).
3. Divide en **80 % entrenamiento / 20 % prueba** con `random_state=412` y `stratify=y` (mantiene la proporción de las 7 clases).
4. **Imputación con `SimpleImputer`**: mediana para numéricas y moda para categóricas. Se ajusta solo con train.
5. **One-Hot Encoding** con `pd.get_dummies(drop_first=True)`. Train y test quedan con las mismas columnas y en el mismo orden.
6. **Escalado con `StandardScaler`** en las 8 columnas numéricas. Se ajusta solo con train, para no filtrar información del test.
7. Exporta `X_train.csv` (1.688 filas), `X_test.csv` (423 filas), `y_train.csv` y `y_test.csv` a `datos/`. Tras el One-Hot quedan **23 columnas**.

## 4. Modelos

### Modelo clásico: Random Forest (`2_modelo_clasico.py`)

```python
RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=2,
                       random_state=412, n_jobs=-1)
```

- 200 árboles de decisión que votan la clase.
- Control de sobreajuste: Accuracy en train 99,47 % frente a 93,85 % en test (diferencia 5,61 %).
- Variables más importantes: `Weight` (31,07 %), `Age` (9,74 %), `FCVC` (9,51 %), `Height` (8,74 %) y `Gender_Male` (6,32 %).

### Red neuronal: TensorFlow/Keras (`3_red_neuronal.py`)

```
Entrada (23) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(7, Softmax)
```

- Optimizador **Adam** (`learning_rate=0.001`).
- Pérdida **`sparse_categorical_crossentropy`**, porque `y` son enteros 0..6.
- **30 épocas**, `validation_split=0.2` y tamaño de lote por defecto de Keras (32).
- Semilla fija con `tf.keras.utils.set_random_seed(412)`: dos ejecuciones dan las mismas métricas.
- La loss de entrenamiento baja de 1,8437 a 0,0664. En la última época, accuracy de entrenamiento 99,19 % y de validación 94,38 %.

Ambos scripts exponen una función que devuelve las mismas claves (`accuracy`, `precision`, `recall`, `f1`, `y_pred`, `tiempo_entrenamiento`):

- `evaluar_modelo_clasico(verbose=True)`
- `evaluar_red_neuronal(verbose=True)`

### Comparación (`4_comparacion.py`)

Importa las dos funciones con `importlib`, recalcula las métricas sobre el mismo `X_test` / `y_test`, imprime la tabla comparativa y escribe la conclusión técnica.

## 5. Instalación y ejecución

Requisitos: Python 3.11 y [uv](https://docs.astral.sh/uv/) (si no lo tienes: `pip install uv`).

```powershell
# 1. Crear el entorno virtual
uv venv --python 3.11

# 2. Activarlo (PowerShell)
.venv\Scripts\activate
# En Git Bash: source .venv/Scripts/activate

# 3. Instalar dependencias (pandas, scikit-learn, tensorflow)
uv pip install -r requirements.txt
```

Ejecutar en este orden:

```powershell
python 1_preprocesamiento.py   # genera los 4 CSV procesados en datos/
python 2_modelo_clasico.py     # Random Forest y sus métricas
python 3_red_neuronal.py       # red neuronal y sus métricas
python 4_comparacion.py        # tabla comparativa y conclusión (demo en vivo)
```

## 6. Resultados (conjunto de prueba, 423 registros)

| Métrica | Random Forest (Scikit-Learn) | Red Neuronal (Keras) |
|---|:-:|:-:|
| Accuracy | 0,9385 | 0,9385 |
| Precision (weighted) | 0,9439 | 0,9396 |
| Recall (weighted) | 0,9385 | 0,9385 |
| F1-Score (weighted) | 0,9397 | 0,9388 |
| Tiempo de entrenamiento | 1,10 s | 12,75 s |

Las métricas son idénticas en cada ejecución. El tiempo cambia según el equipo y la carga del momento (en otra ejecución: 1,58 s y 20,26 s).

## 7. Conclusión

- **Los dos modelos rinden prácticamente igual.** Tienen el mismo Accuracy (93,85 %) y el Random Forest supera a la red por solo 0,09 puntos en F1 y 0,43 en Precision, una diferencia muy pequeña para declarar un ganador claro.
- **Random Forest es la opción más práctica para este dataset:** entrena más de 10 veces más rápido, no necesita elegir épocas ni learning rate, y muestra qué variables pesan más (el peso es la principal).
- **La red neuronal también funciona bien**, gracias al escalado con `StandardScaler`, pero cuesta más entrenarla y es menos interpretable.
- Los modelos coinciden en el 91,49 % de las predicciones de test, así que no siempre se equivocan en los mismos casos.
- Con datos tabulares pequeños como estos (2.111 registros), un modelo clásico basado en árboles es suficiente. Una red neuronal podría ganar ventaja con muchos más datos.

## 8. Estructura del proyecto

```text
machine_lenin/
├── datos/
│   ├── datos_obesidad.csv      # dataset original
│   ├── X_train.csv             # generado por 1_preprocesamiento.py
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── 1_preprocesamiento.py       # imputación, One-Hot, escalado y split 80/20
├── 2_modelo_clasico.py         # Random Forest (Scikit-Learn)
├── 3_red_neuronal.py           # red neuronal (TensorFlow/Keras)
├── 4_comparacion.py            # tabla comparativa y conclusión
├── README.md                   # este documento
├── requirements.txt            # dependencias
└── HOJA_DE_RUTA.md             # plan de trabajo del equipo
```
