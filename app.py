"""Application web Streamlit : prédit si un vin est « bon » (note >= 7).

Lancer avec : streamlit run app.py
"""
from pathlib import Path

import streamlit as st

from src.winequality.predict import load_model, predict_one

MODEL_PATH = Path("models/wine_svm.joblib")

# (min, max, valeur par défaut, pas) d'après les plages du dataset UCI
CHAMPS = {
    "fixed acidity": (3.5, 16.0, 7.0, 0.1),
    "volatile acidity": (0.05, 1.7, 0.3, 0.01),
    "citric acid": (0.0, 1.7, 0.3, 0.01),
    "residual sugar": (0.5, 70.0, 3.0, 0.1),
    "chlorides": (0.01, 0.7, 0.05, 0.001),
    "free sulfur dioxide": (1.0, 300.0, 30.0, 1.0),
    "total sulfur dioxide": (5.0, 450.0, 115.0, 1.0),
    "density": (0.985, 1.04, 0.995, 0.0001),
    "pH": (2.7, 4.0, 3.2, 0.01),
    "sulphates": (0.2, 2.0, 0.5, 0.01),
    "alcohol": (8.0, 15.0, 10.3, 0.1),
}

st.set_page_config(page_title="Qualité du vin", page_icon="🍷")
st.title(" Prédiction de la qualité du vin")
st.write("Renseigne les caractéristiques physico-chimiques du vin : le modèle "
         "SVM prédit s'il est **bon** (note ≥ 7).")


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Modèle introuvable. Lance d'abord : `python -m winequality.train`")
    st.stop()

type_vin = st.radio("Type de vin", ["white", "red"],
                    format_func=lambda t: "Blanc" if t == "white" else "Rouge",
                    horizontal=True)

obs = {}
col1, col2 = st.columns(2)
for i, (nom, (lo, hi, defaut, pas)) in enumerate(CHAMPS.items()):
    with (col1 if i % 2 == 0 else col2):
        obs[nom] = st.number_input(nom, min_value=lo, max_value=hi,
                                   value=defaut, step=pas, format="%.4f"
                                   if pas < 0.01 else "%.2f")
obs["type"] = type_vin

if st.button("Prédire", type="primary"):
    label, indice = predict_one(get_model(), obs)
    if label == 1:
        st.success(f"Vin prédit **bon** (indice de confiance : {indice:.0%})")
    else:
        st.warning(f"Vin prédit **pas bon** (indice de confiance : {indice:.0%} ; 50 % = frontière)")
    st.caption("L'indice n'est pas une probabilité calibrée. Le modèle privilégie le rappel (il repère ~82 % des bons vins) "
               "mais sa précision est d'environ 40 % : à utiliser comme "
               "pré-filtre, pas comme verdict.")
