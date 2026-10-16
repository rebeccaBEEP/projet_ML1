import pytest

from winequality.predict import load_model, predict_one
from winequality.train import train


@pytest.fixture
def modele(data_dir, tmp_path):
    sortie = tmp_path / "modele.joblib"
    train(data_dir, sortie)
    return load_model(sortie)


def test_predict_one_renvoie_label_et_probabilite(modele, vin_exemple):
    label, indice = predict_one(modele, vin_exemple)
    assert label in (0, 1)
    assert (indice > 0.5) == (label == 1)
    assert 0.0 <= indice <= 1.0


def test_type_accepte_texte_ou_entier(modele, vin_exemple):
    avec_texte = predict_one(modele, {**vin_exemple, "type": "red"})
    avec_entier = predict_one(modele, {**vin_exemple, "type": 1})
    assert avec_texte == avec_entier


def test_plus_d_alcool_augmente_l_indice(modele, vin_exemple):
    _, faible = predict_one(modele, {**vin_exemple, "alcohol": 9.0})
    _, fort = predict_one(modele, {**vin_exemple, "alcohol": 13.5})
    assert fort > faible


def test_variable_manquante_leve_une_erreur(modele, vin_exemple):
    del vin_exemple["alcohol"]
    with pytest.raises(ValueError, match="manquantes"):
        predict_one(modele, vin_exemple)


def test_type_invalide_leve_une_erreur(modele, vin_exemple):
    with pytest.raises(ValueError):
        predict_one(modele, {**vin_exemple, "type": "rose"})
