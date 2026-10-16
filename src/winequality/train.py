"""Entraîne le modèle final et le sauvegarde.

Usage : python -m winequality.train --data-dir data --output models/wine_svm.joblib
"""
import argparse
from pathlib import Path

import joblib
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from .model import build_model
from .preprocessing import load_data, preprocess


def train(data_dir="data", output="models/wine_svm.joblib"):
    X, y = preprocess(load_data(data_dir))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = build_model().fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "précision": precision_score(y_test, y_pred),
        "rappel": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, model.decision_function(X_test)),
    }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="models/wine_svm.joblib")
    args = parser.parse_args()

    _, metrics = train(args.data_dir, args.output)
    print("Métriques sur le jeu de test :")
    for nom, valeur in metrics.items():
        print(f"  {nom:10s}: {valeur:.3f}")
    print(f"Modèle sauvegardé dans {args.output}")


if __name__ == "__main__":
    main()
