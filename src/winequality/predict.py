"""Prédiction pour une observation (utilisée par l'application web)."""
import math

import joblib
import pandas as pd

from .preprocessing import FEATURES, TYPE_CODES


def load_model(path="models/wine_svm.joblib"):
    return joblib.load(path)


def predict_one(model, observation):
    """Prédit pour un vin décrit par un dict {nom de variable: valeur}.

    `type` peut valoir "red"/"white" ou 1/0.
    Retourne (label, indice) avec label = 1 (bon) ou 0. L'indice est le score de
    décision du SVM passé dans une sigmoïde : > 0.5 pour « bon ». Ce n'est PAS
    une probabilité calibrée, seulement un indice de confiance.
    """
    obs = dict(observation)
    manquantes = [f for f in FEATURES if f not in obs]
    if manquantes:
        raise ValueError(f"Variables manquantes : {manquantes}")
    if isinstance(obs["type"], str):
        if obs["type"] not in TYPE_CODES:
            raise ValueError("type doit valoir 'red' ou 'white'")
        obs["type"] = TYPE_CODES[obs["type"]]

    X = pd.DataFrame([[obs[f] for f in FEATURES]], columns=FEATURES)
    label = int(model.predict(X)[0])
    score = float(model.decision_function(X)[0])
    return label, 1 / (1 + math.exp(-score))
