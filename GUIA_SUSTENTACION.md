# 🎓 Documento Maestro de Estudio y Sustentación — Taller Final ML 601

> **Proyecto:** Clasificación multiclase de niveles de obesidad (`NObeyesdad`, 7 clases) comparando **Random Forest (Scikit-Learn)** vs **Red Neuronal Secuencial (TensorFlow/Keras)**.  
> **Integrantes:** Andrés Felipe Bonilla Aya & Omar Bravo Montaña  
> **Fecha de entrega y sustentación:** Martes **24 de noviembre de 2026** (hasta las 16:00 h) — **5 a 8 minutos por estudiante** + ejecución en vivo de `python 4_comparacion.py`.

---

## 1. Arquitectura General del Pipeline (De principio a fin)

```text
datos/datos_obesidad.csv (2,111 filas × 17 cols, 0 nulos)
        │
        ▼  [1_preprocesamiento.py]
datos/X_train.csv (1,688 × 23)   datos/X_test.csv (423 × 23)
datos/y_train.csv (1,688 × 1)    datos/y_test.csv (423 × 1)
        │                                │
        ├────────────────┬───────────────┤
        ▼                ▼               │
[2_modelo_clasico.py]  [3_red_neuronal.py]
Random Forest (SKL)    Keras Sequential
        │                │
        └───────┬────────┘
                ▼  [4_comparacion.py]
     Evaluación idéntica sobre X_test / y_test
     Tabla comparativa + Acuerdo (91.49%) + Conclusión
```

### Contrato entre scripts
Tanto `2_modelo_clasico.py` (`evaluar_modelo_clasico`) como `3_red_neuronal.py` (`evaluar_red_neuronal`) pueden ejecutarse solos desde consola **y también** devuelven exactamente el mismo diccionario:
```python
{
    "accuracy": float,
    "precision": float,
    "recall": float,
    "f1": float,
    "y_pred": np.ndarray,           # vector (423,) con enteros 0..6
    "tiempo_entrenamiento": float,  # segundos
}
```
Eso permite que `4_comparacion.py` los invoque en vivo sin duplicar código ni cambiar datos.

---

## 2. Explicación Técnica Detallada de Cada Archivo

### 2.1 `1_preprocesamiento.py` — Qué hace y por qué cada línea importa

1. **Rutas portables (`Path(__file__).resolve().parent / "datos"`):**
   - Garantiza que el script encuentre `datos/` incluso si lo ejecutan desde otra carpeta de la terminal.
2. **Codificación ordinal clínica de `NObeyesdad` (`MAPEO_CLASES`):**
   - Convierte las 7 etiquetas de texto en enteros `0` a `6` en orden clínico progresivo:
     - `0: Insufficient_Weight` (272)
     - `1: Normal_Weight` (287)
     - `2: Overweight_Level_I` (290)
     - `3: Overweight_Level_II` (290)
     - `4: Obesity_Type_I` (351)
     - `5: Obesity_Type_II` (297)
     - `6: Obesity_Type_III` (324)
   - *Ventaja:* En la matriz de confusión, los errores quedan agrupados cerca de la diagonal (por ejemplo confundir nivel 1 con 2 es leve; confundir 0 con 6 sería grave).
3. **División ANTES de transformar (`train_test_split`):**
   - `test_size=0.2` (1,688 train / 423 test), `random_state=412` (exigido por el taller) y `stratify=y` (asegura que cada clase conserve exactamente ~20% en Test).
   - **¿Por qué dividir ANTES de imputar y escalar?** Para evitar **fuga de datos (*data leakage*)**: ni la mediana, ni la moda, ni la media/desviación de `StandardScaler` deben "ver" filas de `X_test`.
4. **`SimpleImputer` (mediana para numéricas, moda para categóricas):**
   - El dataset tiene **0% de nulos**, así que no modifica valores, pero se incluye ajustado sobre `X_train` (`fit_transform`) y aplicado a `X_test` (`transform`) para dejar el pipeline robusto ante futuros datos incompletos y cumplir el requisito técnico del taller.
5. **One-Hot Encoding consistente (`pd.Categorical` + `pd.get_dummies(..., drop_first=True, dtype=int)`):**
   - Convierte las 8 categóricas en variables binarias `0/1`, pasando de **16** a **23 columnas finales**.
   - **`drop_first=True`:** Elimina la primera categoría de cada variable como referencia base para evitar **colinealidad perfecta** (*dummy variable trap*), especialmente importante si se usan modelos lineales o redes.
   - **`pd.Categorical(..., categories=categorias_conocidas)` + `X_test.reindex(...)`:** En este dataset, `CALC == 'Always'` aparece **1 sola vez** en las 2,111 filas. Si quedara solo en Train o solo en Test, un `get_dummies` normal produciría columnas distintas entre Train y Test y fallaría el modelo. Fijar las categorías conocidas previene ese fallo.
   - **`dtype=int`:** En Pandas 3.x, `get_dummies` devuelve booleanos (`True/False`) por defecto; forzar `int` asegura compatibilidad directa con TensorFlow.
6. **`StandardScaler` solo sobre las 8 numéricas:**
   - Transforma `Age`, `Height`, `Weight`, `FCVC`, `NCP`, `CH2O`, `FAF`, `TUE` a media `0.0` y desviación estándar `1.0` (`z = (x - μ) / σ`).
   - *Por qué no escalar las columnas One-Hot:* ya están en `{0, 1}`; escalarlas distorsionaría su interpretación binaria sin beneficiar a los modelos.

---

### 2.2 `2_modelo_clasico.py` — Random Forest (Scikit-Learn)

1. **Modelo elegido:**
   ```python
   RandomForestClassifier(
       n_estimators=200,
       max_depth=12,
       min_samples_leaf=2,
       random_state=412,
       n_jobs=-1,
   )
   ```
   - **`n_estimators=200`:** Construye 200 árboles de decisión entrenados sobre submuestras *bootstrap* y subconjuntos aleatorios de variables; la predicción final es por votación.
   - **`max_depth=12` y `min_samples_leaf=2`:** Regularización explícita para evitar que los árboles memoricen registros individuales (sin límite de profundidad suelen memorizar al 100% el entrenamiento).
   - **`random_state=412`:** Reproducibilidad exacta.
   - **`n_jobs=-1`:** Paraleliza el entrenamiento usando todos los núcleos del procesador (~`0.75–1.96 s`).
2. **Control de sobreajuste (*Overfitting*):**
   - Accuracy en Train: **`99.47%`** vs Accuracy en Test: **`93.85%`** → brecha de **`5.61%`**, generalización sólida.
3. **Importancia de variables (`feature_importances_`):**
   - `Weight` (**31.07%**), `Age` (**9.74%**), `FCVC` (**9.51%** — consumo de verduras), `Height` (**8.74%**), `Gender_Male` (**6.32%**).
   - Tiene total sentido clínico: la definición de obesidad depende directamente de peso y estatura (IMC), edad y hábitos alimenticios.
4. **Lectura de la matriz de confusión (423 muestras de Test):**
   - `Obesity_Type_III`: **65/65** (100% precisión y recall).
   - `Obesity_Type_II`: **59/60** (98.33% recall).
   - Las confusiones ocurren casi exclusivamente entre clases vecinas de peso normal / sobrepeso (`Normal_Weight` tiene precision `77.14%` porque algunos casos de `Insufficient_Weight`, `Overweight_Level_I` y `Overweight_Level_II` quedan en la frontera).

---

### 2.3 `3_red_neuronal.py` — Red Neuronal Secuencial (TensorFlow/Keras)

1. **Semilla dentro de la función (`tf.keras.utils.set_random_seed(412)`):**
   - Fija las semillas de Python, NumPy y TensorFlow justo antes de crear la red, garantizando que `4_comparacion.py` obtenga exactamente los mismos resultados que `3_red_neuronal.py` ejecutado solo.
2. **Conversión `.values.astype("float32")` y `.values.ravel()`:**
   - Keras trabaja nativamente en coma flotante de 32 bits (`float32`), más rápido y eficiente en memoria que `float64`.
   - `ravel()` convierte `y_train` de forma `(1688, 1)` a vector 1D `(1688,)`.
3. **Arquitectura `Sequential`:**
   - `Input(shape=(23,))`: recibe las 23 características preprocesadas.
   - `Dense(64, activation="relu")`: primera capa oculta; **ReLU** ($\max(0, x)$) introduce no linealidad sin sufrir desvanecimiento del gradiente como Sigmoid/Tanh en capas ocultas.
   - `Dense(32, activation="relu")`: segunda capa oculta que comprime las representaciones antes de clasificar.
   - `Dense(7, activation="softmax")`: 7 neuronas (una por clase); **Softmax** convierte los *logits* en una distribución de probabilidad donde las 7 salidas son positivas y suman **`1.0`**.
4. **Compilación y entrenamiento:**
   - `Adam(learning_rate=0.001)`: optimizador adaptativo estándar por su rápida convergencia.
   - `loss="sparse_categorical_crossentropy"`: función de pérdida para clasificación multiclase cuando `y` viene como enteros `0..6` (si `y` fuera una matriz One-Hot de 7 columnas, se usaría `categorical_crossentropy`).
   - `epochs=30`, `validation_split=0.2`: reserva el 20% de `X_train` (~338 filas) para monitorear `val_loss` y `val_accuracy` en cada época **sin tocar `X_test`**.
   - Evolución observada:
     - Época 1: `loss=1.8437`, `accuracy=0.2711`, `val_loss=1.6554`, `val_accuracy=0.4231`.
     - Época 30: `loss=0.0664`, `accuracy=0.9919`, `val_loss=0.1541`, `val_accuracy=0.9438`.
5. **Evaluación en Test:**
   - `modelo.evaluate(X_test, y_test)` → `loss = 0.2100`.
   - `np.argmax(modelo.predict(X_test), axis=1)` → toma la clase con mayor probabilidad para cada fila y calcula Accuracy (`0.9385`), Precision (`0.9396`), Recall (`0.9385`) y F1-Score (`0.9388`) con `average="weighted"`.

---

### 2.4 `4_comparacion.py` — Integración y Comparación Final

1. **`os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")`:**
   - Debe ir *antes* de importar TensorFlow (a través de `3_red_neuronal`) para ocultar mensajes informativos `INFO` de C++ y dejar la tabla limpia en la demo en vivo.
2. **`importlib.import_module("2_modelo_clasico")`:**
   - En Python no se puede escribir `import 2_modelo_clasico` porque los nombres de módulos no pueden empezar por número; `importlib.import_module` resuelve esa importación dinámicamente manteniendo los nombres numéricos exigidos por el taller.
3. **Resultados comparativos exactos sobre las 423 muestras de `X_test`:**

| Métrica (Test Set) | Random Forest (Scikit-Learn) | Red Neuronal (Keras) |
| :--- | :---: | :---: |
| **Accuracy** | **93.85%** (`0.9385`) | **93.85%** (`0.9385`) |
| **Precision (weighted)** | **94.39%** (`0.9439`) | 93.96% (`0.9396`) |
| **Recall (weighted)** | **93.85%** (`0.9385`) | **93.85%** (`0.9385`) |
| **F1-Score (weighted)** | **93.97%** (`0.9397`) | 93.88% (`0.9388`) |
| **Tiempo de entrenamiento** | **~0.75–1.96 s** | ~8.89–12.75 s |

- **Acuerdo entre predicciones:** `91.49%` (aciertan las mismas 397/423 en total, pero no siempre en los mismos individuos).

---

## 3. Guion Sugerido para la Sustentación (5–8 min por estudiante)

> [!TIP]
> **Truco para la demo en vivo:** Al iniciar su turno (o justo al terminar la Slide 4), abran la terminal en la raíz del proyecto con `.venv` activado y ejecuten `python 4_comparacion.py`. Mientras TensorFlow termina de cargar (~10–20 s), explican la Slide 5 (*qué hace el script y cómo leer la tabla*) y cuando muestre la tabla ya tienen los datos frescos en pantalla.

| Tiempo | Quién habla | Diapositivas | Qué explicar |
| :--- | :--- | :--- | :--- |
| **0:00 – 3:00** | **Integrante B (Omar)** | Slides 1, 2, 3 | Presentación, pregunta de negocio, dataset UCI (2,111 filas, 16 features, 7 clases balanceadas, 0 nulos) y los 4 pasos de `1_preprocesamiento.py` sin *data leakage*. |
| **3:00 – 5:30** | **Integrante B (Omar)** | Slide 4 (parte NN) | Arquitectura de la Red Neuronal (`23 → 64 ReLU → 32 ReLU → 7 Softmax`), `Adam`, `sparse_categorical_crossentropy`, 30 épocas y descenso del `loss`. |
| **5:30 – 8:00** | **Integrante A (Andrés)** | Slide 4 (RF) + Slide 5 | Modelo Random Forest (200 árboles, `max_depth=12`) y **ejecución en vivo de `python 4_comparacion.py`** explicando la tabla. |
| **8:00 – 11:30** | **Integrante A (Andrés)** | Slides 6, 7, 8 | Empate técnico en Accuracy (`93.85%`), ventaja en velocidad de Random Forest, importancia de variables (`Weight` 31.07%), análisis de overfitting y respuesta final a la pregunta de negocio. |

*(También pueden repartirse 6–7 min cada uno si el docente da 5–8 min por persona: por ejemplo Integrante 1 explica Datos + Preprocesamiento + Random Forest, e Integrante 2 explica Red Neuronal + Demo en vivo + Comparación y Conclusiones).*

---

## 4. Banco de Preguntas del Docente (Código y Conceptos)

### A. Preprocesamiento y Datos
1. **¿Por qué incluyeron `SimpleImputer` si el dataset tiene 0% de nulos?**  
   *Respuesta:* Por requisito de arquitectura del pipeline y buena práctica de producción: si en el futuro llegan nuevos pacientes con datos faltantes, el flujo ya sabe imputar numéricas con la mediana (robusta a outliers) y categóricas con la moda, usando siempre estadísticas aprendidas solo en `X_train`.
2. **¿Por qué `StandardScaler` y no `MinMaxScaler`?**  
   *Respuesta:* `StandardScaler` centra en media 0 y desviación 1, es menos sensible a valores extremos que `MinMaxScaler` (que comprime todo entre 0 y 1 si hay un outlier) y ayuda a que el gradiente de `Adam` en la red neuronal converja de forma estable y rápida.
3. **¿Por qué usaron `stratify=y` en `train_test_split`?**  
   *Respuesta:* Porque es clasificación multiclase con 7 niveles. `stratify=y` garantiza que la proporción de las 7 clases sea idéntica (~80/20) en Train y en Test, evitando sesgos por azar en alguna clase.
4. **¿Para qué sirve `drop_first=True` en `pd.get_dummies` y por qué usaron `pd.Categorical` + `reindex`?**  
   *Respuesta:* `drop_first=True` elimina una categoría redundante por variable para evitar colinealidad perfecta. `pd.Categorical` y `reindex` aseguran que `X_train` y `X_test` tengan exactamente las mismas 23 columnas aunque una categoría rara (como `CALC='Always'`, que solo tiene 1 registro) aparezca en un solo subconjunto.

### B. Modelo Clásico (Scikit-Learn)
5. **¿Por qué Random Forest y no un solo Árbol de Decisión?**  
   *Respuesta:* Un solo árbol tiende a sobreajustar fácilmente. Random Forest promedia 200 árboles entrenados con *bagging* y selección aleatoria de variables, reduciendo la varianza y mejorando la generalización en Test.
6. **¿Para qué pusieron `max_depth=12` y `min_samples_leaf=2`?**  
   *Respuesta:* Para controlar el *overfitting*. Sin límite de profundidad, el árbol memoriza el 100% del Train; con estos límites logramos `99.47%` en Train y `93.85%` en Test (solo `5.61%` de diferencia).
7. **¿Por qué usan `average="weighted"` en Precision, Recall y F1?**  
   *Respuesta:* En clasificación multiclase hay una métrica por cada una de las 7 clases. `weighted` calcula el promedio ponderado por la cantidad real de ejemplos (`support`) de cada clase en `y_test`.
8. **¿Qué variables dominan y por qué?**  
   *Respuesta:* `Weight` (`31.07%`), `Age` (`9.74%`), `FCVC` (`9.51%`), `Height` (`8.74%`) y `Gender_Male` (`6.32%`). Tiene sentido fisiológico porque las categorías de obesidad están directamente relacionadas con el índice de masa corporal (peso y estatura) y hábitos diarios.

### C. Red Neuronal (TensorFlow / Keras)
9. **¿Por qué `ReLU` en las capas ocultas y `Softmax` en la salida?**  
   *Respuesta:* `ReLU` evita que los gradientes se anulen durante la retropropagación y es muy rápida de calcular. `Softmax` en la capa de 7 neuronas convierte las salidas en probabilidades positivas que suman `1.0`, ideal para clasificación multiclase excluyente (una persona pertenece a un solo nivel).
10. **¿Por qué `sparse_categorical_crossentropy` y no `categorical_crossentropy` o `binary_crossentropy`?**  
    *Respuesta:* `binary_crossentropy` es solo para 2 clases. Entre las categóricas, se usa `sparse_categorical_crossentropy` porque nuestras etiquetas `y` son números enteros (`0..6`), mientras que `categorical_crossentropy` exigiría codificar `y` en vectores One-Hot de 7 columnas.
11. **¿Cuál es la diferencia entre `validation_split=0.2` dentro de `fit()` y `X_test`?**  
    *Respuesta:* `validation_split=0.2` toma un 20% de `X_train` durante el entrenamiento para monitorear época a época si la red empieza a sobreajustar (`val_loss`). `X_test` (las 423 muestras) nunca participa en el entrenamiento y se usa únicamente al final para medir el desempeño real sobre datos nunca vistos.
12. **¿Cómo saben que la red neuronal convergió bien y no memorizó?**  
    *Respuesta:* Porque tanto `loss` (`1.8437 → 0.0664`) como `val_loss` (`1.6554 → 0.1541`) descendieron de manera continua durante las 30 épocas sin dispararse hacia arriba, y en `X_test` obtuvo `93.85%` de Accuracy (muy cercano al `94.38%` de validación).
13. **¿Por qué llamaron `tf.keras.utils.set_random_seed(412)` dentro de `evaluar_red_neuronal()`?**  
    *Respuesta:* Si se pusiera solo al inicio del archivo, cuando `4_comparacion.py` ejecuta primero otro código, el estado aleatorio cambiaría. Ponerla dentro de la función garantiza que los pesos iniciales y el barajado de lotes sean idénticos cada vez que se llama.

### D. Código Python y Comparación Final
14. **¿Por qué en `4_comparacion.py` usaron `importlib.import_module("2_modelo_clasico")` en vez de `import 2_modelo_clasico`?**  
    *Respuesta:* Porque la sintaxis de Python prohíbe que un identificador empiece por un dígito. Como la guía exige nombrar los archivos `1_`, `2_`, `3_`, `4_`, `importlib.import_module` permite importarlos limpiamente por su nombre en cadena.
15. **¿Si los dos empatan en `93.85%`, cuál recomiendan y por qué?**  
    *Respuesta:* Para este dataset tabular de 2,111 registros recomendamos **Random Forest**, porque alcanza el mismo `93.85%` de Accuracy, un F1 ligeramente superior (`93.97%` vs `93.88%`), entrena ~10 veces más rápido (~1 s vs ~9 s) y ofrece explicabilidad directa mediante la importancia de variables. La Red Neuronal sigue siendo valiosa y escalaría mejor si el dataset creciera a cientos de miles de registros o incluyera señales no tabulares.
