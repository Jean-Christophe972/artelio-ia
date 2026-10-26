"""Paramètres du service IA. Les secrets viennent de l'environnement (.env non versionné)."""

from __future__ import annotations

import os
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
DOSSIER_DATA = Path(os.environ.get("ARTELIO_DATA_DIR", RACINE / "data"))
DOSSIER_MODELES = Path(os.environ.get("ARTELIO_MODELS_DIR", RACINE / "models"))

NOM_MODELE = "artelio-qualifieur"  # nom dans le registre MLflow
GRAINE = 42  # reproductibilité : même données + même graine = même modèle

METIERS = ["chauffage", "electricite", "menuiserie", "peinture", "petits_travaux", "plomberie", "serrurerie"]
URGENCES = ["faible", "normale", "haute", "critique"]  # ordre croissant de gravité

# --- Politique de décision (plan de gouvernance §8 : supervision humaine) ----------------
# Seuils fixés a priori par le métier, puis MESURÉS (et non optimisés) sur le jeu de test :
#   - en dessous de SEUIL_CONFIANCE_METIER, la demande part en revue humaine ;
#   - dès que P(critique) dépasse SEUIL_ALERTE_CRITIQUE, on traite comme critique
#     (une fausse alerte coûte moins cher qu'une fuite d'eau ignorée).
SEUIL_CONFIANCE_METIER = 0.55
SEUIL_ALERTE_CRITIQUE = 0.25
#   - au-delà de SEUIL_HORS_DOMAINE de mots inconnus du vocabulaire, le texte est probablement hors
#     sujet (« quel temps fera-t-il demain ? ») : un classifieur y est souvent trop sûr de lui.
SEUIL_HORS_DOMAINE = 0.5

# --- Barrière de promotion d'un nouveau modèle (champion / challenger) --------------------
TOLERANCE_REGRESSION = 0.01  # le challenger peut perdre au plus 1 point de F1 macro
RAPPEL_CRITIQUE_MIN = 0.85  # au moins 85 % des urgences critiques détectées
MACRO_F1_METIER_MIN = 0.70  # plancher absolu sur le jeu hors gabarits

# Enrichissement des données (voir augmentation.py) : exemples par combinaison (métier, urgence)
N_AUGMENTATION = 120

# Fenêtre du jeu de test temporel (les N derniers jours ne servent jamais à l'entraînement)
JOURS_TEST_TEMPOREL = 60


def env(nom: str, defaut: str | None = None) -> str:
    valeur = os.environ.get(nom, defaut)
    if valeur is None or valeur == "":
        raise RuntimeError(f"Variable d'environnement manquante : {nom}")
    return valeur
