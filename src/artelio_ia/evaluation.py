"""Évaluation du modèle : performance, calibration, robustesse, équité, politique de revue humaine."""

from __future__ import annotations

import random
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, recall_score

from artelio_ia.config import GRAINE, METIERS, URGENCES
from artelio_ia.modele import Qualifieur, ordre_urgence


def metriques(qualifieur: Qualifieur, donnees: pd.DataFrame) -> dict:
    """Métriques complètes sur un jeu étiqueté (texte, metier, urgence)."""
    textes = donnees["texte"].tolist()
    preds = qualifieur.predire(textes)
    y_m, y_u = donnees["metier"].to_numpy(), donnees["urgence"].to_numpy()
    p_m = np.array([p.metier for p in preds])
    p_u = np.array([p.urgence for p in preds])
    revue = np.array([p.revue_humaine for p in preds])
    probas_m = qualifieur.modele_metier.predict_proba(textes)

    ecart_urgence = np.array([ordre_urgence(a) - ordre_urgence(b) for a, b in zip(p_u, y_u, strict=True)])
    auto = ~revue
    return {
        "n": len(donnees),
        "metier": {
            "exactitude": accuracy_score(y_m, p_m),
            "f1_macro": f1_score(y_m, p_m, average="macro"),
            "log_loss": log_loss(y_m, probas_m, labels=list(qualifieur.modele_metier.classes_)),
            "ece": erreur_calibration(probas_m, y_m, list(qualifieur.modele_metier.classes_)),
            "par_classe": classification_report(y_m, p_m, labels=METIERS, output_dict=True, zero_division=0),
            "matrice_confusion": confusion_matrix(y_m, p_m, labels=METIERS).tolist(),
        },
        "urgence": {
            "exactitude": accuracy_score(y_u, p_u),
            "f1_macro": f1_score(y_u, p_u, average="macro"),
            # Le chiffre clé métier : part des urgences critiques effectivement signalées comme critiques
            "rappel_critique": recall_score(y_u, p_u, labels=["critique"], average="macro", zero_division=0),
            # Erreur grave : urgence sous-estimée de 2 niveaux ou plus (ex. critique → normale)
            "sous_estimation_grave": float(np.mean(ecart_urgence <= -2)),
            "par_classe": classification_report(y_u, p_u, labels=URGENCES, output_dict=True, zero_division=0),
            "matrice_confusion": confusion_matrix(y_u, p_u, labels=URGENCES).tolist(),
        },
        "revue_humaine": {
            # Couverture : part des demandes traitées sans humain ; précision sur cette part
            "taux": float(np.mean(revue)),
            "exactitude_metier_automatique": float(np.mean(p_m[auto] == y_m[auto])) if auto.any() else None,
            "exactitude_metier_en_revue": float(np.mean(p_m[revue] == y_m[revue])) if revue.any() else None,
        },
    }


def erreur_calibration(probas: np.ndarray, y: np.ndarray, classes: list[str], n_tranches: int = 10) -> float:
    """ECE : écart moyen entre la confiance annoncée et l'exactitude réelle (0 = parfaitement calibré)."""
    confiance = probas.max(axis=1)
    correct = np.array(classes)[probas.argmax(axis=1)] == y
    tranches = np.linspace(0, 1, n_tranches + 1)
    ece = 0.0
    for bas, haut in zip(tranches[:-1], tranches[1:], strict=True):
        dans = (confiance > bas) & (confiance <= haut)
        if dans.any():
            ece += dans.mean() * abs(correct[dans].mean() - confiance[dans].mean())
    return float(ece)


def par_tranche(qualifieur: Qualifieur, donnees: pd.DataFrame, colonne: str) -> pd.DataFrame:
    """Performance par sous-groupe (ex. canal) : un groupe ne doit pas être nettement moins bien servi."""
    lignes = []
    for valeur, groupe in donnees.groupby(colonne):
        preds = qualifieur.predire(groupe["texte"].tolist())
        lignes.append(
            {
                colonne: valeur,
                "n": len(groupe),
                "exactitude_metier": np.mean([p.metier == m for p, m in zip(preds, groupe["metier"], strict=True)]),
                "exactitude_urgence": np.mean([p.urgence == u for p, u in zip(preds, groupe["urgence"], strict=True)]),
                "taux_revue_humaine": np.mean([p.revue_humaine for p in preds]),
            }
        )
    return pd.DataFrame(lignes)


# --- Robustesse : le modèle doit résister aux textes « abîmés » --------------------------------
def _sans_accents(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def _fautes(t: str, rng: random.Random, taux: float = 0.06) -> str:
    lettres = list(t)
    for i in range(len(lettres) - 1):
        if lettres[i].isalpha() and rng.random() < taux:
            lettres[i], lettres[i + 1] = lettres[i + 1], lettres[i]
    return "".join(lettres)


PERTURBATIONS = {
    "original": lambda t, rng: t,
    "sans_accents": lambda t, rng: _sans_accents(t),
    "majuscules": lambda t, rng: t.upper(),
    "fautes_de_frappe": lambda t, rng: _fautes(t, rng),
    "telephone_ajoute": lambda t, rng: t + " rappelez moi au 06 12 34 56 78",
    "debut_seulement": lambda t, rng: " ".join(t.split()[:5]),
    "bruit_politesse": lambda t, rng: "Bonjour madame monsieur, j'espère que vous allez bien. " + t + " Merci beaucoup, bonne journée.",
}


def robustesse(qualifieur: Qualifieur, donnees: pd.DataFrame) -> pd.DataFrame:
    """Exactitude sous perturbations contrôlées (graine fixe → résultats reproductibles)."""
    lignes = []
    for nom, perturber in PERTURBATIONS.items():
        rng = random.Random(GRAINE)
        textes = [perturber(t, rng) for t in donnees["texte"]]
        preds = qualifieur.predire(textes)
        lignes.append(
            {
                "perturbation": nom,
                "exactitude_metier": np.mean([p.metier == m for p, m in zip(preds, donnees["metier"], strict=True)]),
                "exactitude_urgence": np.mean([p.urgence == u for p, u in zip(preds, donnees["urgence"], strict=True)]),
                "taux_revue_humaine": np.mean([p.revue_humaine for p in preds]),
            }
        )
    return pd.DataFrame(lignes)


SEUILS_ETUDIES = np.arange(0.3, 0.95, 0.05)


def compromis_couverture(qualifieur: Qualifieur, donnees: pd.DataFrame, seuils=SEUILS_ETUDIES) -> pd.DataFrame:
    """Pour chaque seuil de confiance : part traitée automatiquement et exactitude sur cette part."""
    probas = qualifieur.modele_metier.predict_proba(donnees["texte"].tolist())
    classes = np.array(qualifieur.modele_metier.classes_)
    confiance, pred = probas.max(axis=1), classes[probas.argmax(axis=1)]
    correct = pred == donnees["metier"].to_numpy()
    return pd.DataFrame(
        [
            {
                "seuil": round(float(s), 2),
                "couverture_automatique": float(np.mean(confiance >= s)),
                "exactitude_automatique": float(correct[confiance >= s].mean()) if (confiance >= s).any() else np.nan,
            }
            for s in seuils
        ]
    )
