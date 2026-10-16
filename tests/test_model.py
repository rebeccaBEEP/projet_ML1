import joblib
import pytest

from winequality.model import build_model
from winequality.preprocessing import load_data, preprocess
from winequality.train import train


def test_build_model_hyperparametres():
    svm = build_model().named_steps["svm"]
    assert svm.C == 100
    assert svm.gamma == 0.01
    assert svm.class_weight == "balanced"


def test_train_sauvegarde_le_modele_et_renvoie_des_metriques(data_dir, tmp_path):
    sortie = tmp_path / "modele.joblib"
    _, metrics = train(data_dir, sortie)
    assert sortie.exists()
    assert set(metrics) == {"accuracy", "précision", "rappel", "F1", "AUC"}
    assert all(0 <= v <= 1 for v in metrics.values())
    # les données synthétiques sont informatives (alcool) : mieux que le hasard
    assert metrics["AUC"] > 0.6


def test_le_modele_sauvegarde_predit_0_ou_1(data_dir, tmp_path):
    sortie = tmp_path / "modele.joblib"
    train(data_dir, sortie)
    model = joblib.load(sortie)
    X, _ = preprocess(load_data(data_dir))
    assert set(model.predict(X.head(20))) <= {0, 1}
