"""Chargement et prétraitement des données (même logique que le notebook)."""
from pathlib import Path

import pandas as pd

# Ordre des colonnes identique à celui du notebook (X = df sans quality ni bon)
FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "type",  # 0 = blanc, 1 = rouge
]
SEUIL_BON = 7  # un vin est « bon » si quality >= 7
TYPE_CODES = {"white": 0, "red": 1}


def load_data(data_dir="data"):
    """Lit les deux CSV (rouge, blanc) et les concatène avec une colonne `type`."""
    data_dir = Path(data_dir)
    red = pd.read_csv(data_dir / "winequality-red.csv", sep=";")
    white = pd.read_csv(data_dir / "winequality-white.csv", sep=";")
    red["type"] = "red"
    white["type"] = "white"
    return pd.concat([red, white], ignore_index=True)


def make_target(df):
    """Ajoute la cible binaire `bon` (1 si quality >= 7)."""
    df = df.copy()
    df["bon"] = (df["quality"] >= SEUIL_BON).astype(int)
    return df


def preprocess(df):
    """Supprime les doublons, encode `type` en 0/1 et crée la cible `bon`.

    Retourne (X, y) avec X ordonné selon FEATURES.
    """
    df = df.drop_duplicates().copy()
    if not pd.api.types.is_numeric_dtype(df["type"]):
        df["type"] = df["type"].map(TYPE_CODES)
    df = make_target(df)
    return df[FEATURES], df["bon"]
