"""Données synthétiques : les tests ne dépendent pas des vrais CSV."""
import numpy as np
import pandas as pd
import pytest

COLONNES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]


def _faux_vins(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "fixed acidity": rng.normal(7, 1, n),
        "volatile acidity": rng.normal(0.3, 0.1, n),
        "citric acid": rng.normal(0.3, 0.1, n),
        "residual sugar": rng.normal(5, 2, n),
        "chlorides": rng.normal(0.05, 0.01, n),
        "free sulfur dioxide": rng.normal(30, 10, n),
        "total sulfur dioxide": rng.normal(115, 30, n),
        "density": rng.normal(0.995, 0.002, n),
        "pH": rng.normal(3.2, 0.15, n),
        "sulphates": rng.normal(0.5, 0.1, n),
        "alcohol": rng.normal(10.5, 1.2, n),
    })
    # la qualité dépend surtout de l'alcool (comme dans le vrai jeu de données)
    df["quality"] = np.clip(np.round(5 + (df["alcohol"] - 10.5) + rng.normal(0, 0.7, n)), 3, 9).astype(int)
    return df


@pytest.fixture
def data_dir(tmp_path):
    _faux_vins(300, 0).to_csv(tmp_path / "winequality-red.csv", sep=";", index=False)
    _faux_vins(400, 1).to_csv(tmp_path / "winequality-white.csv", sep=";", index=False)
    return tmp_path


@pytest.fixture
def vin_exemple():
    return {
        "fixed acidity": 7.0, "volatile acidity": 0.3, "citric acid": 0.3,
        "residual sugar": 5.0, "chlorides": 0.05, "free sulfur dioxide": 30.0,
        "total sulfur dioxide": 115.0, "density": 0.995, "pH": 3.2,
        "sulphates": 0.5, "alcohol": 11.0, "type": "white",
    }
