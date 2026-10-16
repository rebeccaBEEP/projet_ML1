

## Auteurs

REBECCA KOUADIO
FATOU KA 
MAROUANE NOUARA


# Classification de la qualité du vin

Projet de Machine Learning I : prédire si un vin est **bon** (note ≥ 7) à partir de ses caractéristiques physico-chimiques (dataset UCI *Wine Quality*, vins rouges et blancs portugais, Vinho Verde).

## Résultat

Cinq modèles comparés (régression logistique, k-NN, arbre de décision, Naive Bayes, SVM), optimisés par Grid Search / Random Search avec validation croisée à 5 folds.

**Modèle final : SVM à noyau RBF** (`C=100`, `gamma=0.01`, `class_weight="balanced"`).
Sur le jeu de test : F1 = 0,54, AUC = 0,85, rappel = 0,82, précision = 0,40.
Le modèle repère la plupart des bons vins mais fait beaucoup de fausses alertes : c'est un outil de pré-filtrage, pas un verdict.

## Structure du dépôt

```
projet-vin-ml/
├── data/                  # winequality-red.csv, winequality-white.csv
├── projet_vin.ipynb       # notebook : analyse, modélisation, optimisation
├── rapport/               # rapport technique (PDF + LaTeX)
├── src/winequality/       # code réutilisable extrait du notebook
│   ├── preprocessing.py   #   chargement, doublons, encodage, cible binaire
│   ├── model.py           #   pipeline StandardScaler + SVM
│   ├── train.py           #   entraînement + sauvegarde du modèle
│   └── predict.py         #   prédiction pour une observation
├── app.py                 # application web (Streamlit)
├── tests/                 # tests unitaires (pytest)
├── Dockerfile
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell) ; source .venv/bin/activate sous Linux/macOS
pip install -r requirements-dev.txt
pip install -e .
```

## Utilisation

**1. Entraîner et sauvegarder le modèle**
```bash
python -m winequality.train
```
Affiche les métriques sur le jeu de test et crée `models/wine_svm.joblib`.

**2. Lancer l'application web**
```bash
streamlit run app.py
```
Ouvrir http://localhost:8501, saisir les caractéristiques d'un vin et cliquer sur « Prédire ».

**3. Lancer les tests unitaires**
```bash
pytest
```
Les tests utilisent des données synthétiques : ils n'ont pas besoin des CSV.

## Docker

```bash
docker build -t wine-quality .
docker run -p 8501:8501 wine-quality
```
L'image entraîne le modèle à la construction (à partir de `data/`), puis lance l'application sur http://localhost:8501.

## Intégration continue

Un workflow GitHub Actions (`.github/workflows/tests.yml`) lance `pytest` à chaque push.


