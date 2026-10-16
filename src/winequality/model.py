"""Modèle final : SVM à noyau RBF (hyperparamètres trouvés dans le notebook)."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model(C=100, gamma=0.01):
    """Pipeline StandardScaler + SVM.

    Le scaler fait partie du pipeline : il est appris sur le train uniquement
    et réappliqué automatiquement à toute nouvelle observation.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced",
                    random_state=42)),
    ])
