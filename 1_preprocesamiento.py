from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "datos"
RUTA_DATASET = DATOS_DIR / "datos_obesidad.csv"

COLUMNA_OBJETIVO = "NObeyesdad"
RANDOM_STATE = 412
TEST_SIZE = 0.2

# Mapeo ordinal clínico de la variable objetivo multiclase (0 a 6)
MAPEO_CLASES = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Overweight_Level_I": 2,
    "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,
    "Obesity_Type_II": 5,
    "Obesity_Type_III": 6,
}


def ejecutar_preprocesamiento(
    ruta_csv: Path = RUTA_DATASET,
    directorio_salida: Path = DATOS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Pipeline completo de preprocesamiento para el dataset de obesidad:
    1. Carga datos_obesidad.csv desde datos/.
    2. Separa X (16 variables) e y ('NObeyesdad' codificada de 0 a 6).
    3. Divide en Train (80%) y Test (20%) estratificado con random_state=412.
    4. Imputa nulos ajustando solo sobre Train (mediana numéricos, moda categóricos).
    5. Aplica One-Hot Encoding consistente a variables categóricas.
    6. Escala variables numéricas con StandardScaler ajustado solo sobre Train.
    7. Exporta X_train.csv, X_test.csv, y_train.csv e y_test.csv en datos/.
    """
    print("=" * 72)
    print("1. CARGANDO DATASET ORIGINAL")
    print("=" * 72)
    df = pd.read_csv(ruta_csv)
    print(f"Archivo cargado : {ruta_csv.name}")
    print(f"Dimensiones     : {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"Nulos totales   : {int(df.isna().sum().sum())}")

    print("\n" + "=" * 72)
    print("2. SEPARANDO CARACTERÍSTICAS (X) Y VARIABLE OBJETIVO (y)")
    print("=" * 72)
    if COLUMNA_OBJETIVO not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{COLUMNA_OBJETIVO}'.")

    clases_desconocidas = set(df[COLUMNA_OBJETIVO].unique()) - set(MAPEO_CLASES.keys())
    if clases_desconocidas:
        raise ValueError(f"Clases no esperadas en '{COLUMNA_OBJETIVO}': {clases_desconocidas}")

    X = df.drop(columns=[COLUMNA_OBJETIVO]).copy()
    y = df[COLUMNA_OBJETIVO].map(MAPEO_CLASES).astype(int)

    num_cols = list(X.select_dtypes(include=["number"]).columns)
    cat_cols = list(X.select_dtypes(exclude=["number", "bool"]).columns)

    # Registrar categorías conocidas para garantizar columnas One-Hot idénticas en Train y Test
    categorias_conocidas = {
        col: sorted(X[col].dropna().astype(str).unique().tolist())
        for col in cat_cols
    }

    print(f"Variables de entrada : {X.shape[1]} ({len(num_cols)} numéricas, {len(cat_cols)} categóricas)")
    print(f"Numéricas            : {num_cols}")
    print(f"Categóricas          : {cat_cols}")
    print("\nMapeo de clases objetivo (NObeyesdad):")
    for etiqueta, codigo in MAPEO_CLASES.items():
        cantidad = int((y == codigo).sum())
        print(f"  {codigo} -> {etiqueta:<20} ({cantidad} registros)")

    print("\n" + "=" * 72)
    print(f"3. DIVISIÓN TRAIN / TEST (80% / 20%, random_state={RANDOM_STATE}, estratificado)")
    print("=" * 72)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train = X_train.copy()
    X_test = X_test.copy()
    print(f"X_train inicial: {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_test  inicial: {X_test.shape}  | y_test : {y_test.shape}")

    print("\n" + "=" * 72)
    print("4. IMPUTACIÓN DE VALORES NULOS (SimpleImputer: mediana / moda)")
    print("=" * 72)
    if num_cols:
        imputer_num = SimpleImputer(strategy="median")
        X_train[num_cols] = imputer_num.fit_transform(X_train[num_cols])
        X_test[num_cols] = imputer_num.transform(X_test[num_cols])
    if cat_cols:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        X_train[cat_cols] = imputer_cat.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = imputer_cat.transform(X_test[cat_cols])
    print("Imputación aplicada correctamente (ajustada solo sobre Train).")

    print("\n" + "=" * 72)
    print("5. CODIFICACIÓN ONE-HOT Y ESCALADO NUMÉRICO (StandardScaler)")
    print("=" * 72)
    for col in cat_cols:
        X_train[col] = pd.Categorical(X_train[col].astype(str), categories=categorias_conocidas[col])
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=categorias_conocidas[col])

    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True, dtype=int)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    print(f"Dimensiones tras One-Hot Encoding -> X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Total de columnas finales         : {X_train.shape[1]}")

    print("\n" + "=" * 72)
    print("6. EXPORTANDO ARCHIVOS PROCESADOS A datos/")
    print("=" * 72)
    directorio_salida.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(directorio_salida / "X_train.csv", index=False)
    X_test.to_csv(directorio_salida / "X_test.csv", index=False)
    y_train.to_frame(name=COLUMNA_OBJETIVO).to_csv(directorio_salida / "y_train.csv", index=False)
    y_test.to_frame(name=COLUMNA_OBJETIVO).to_csv(directorio_salida / "y_test.csv", index=False)

    for nombre in ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"):
        print(f"  [OK] {directorio_salida / nombre}")
    print("=" * 72)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    ejecutar_preprocesamiento()
