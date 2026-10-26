"""Métriques, calibration, robustesse et barrière de promotion."""

import numpy as np
import pandas as pd

from artelio_ia import evaluation as ev
from artelio_ia.entrainement import decider_promotion


def test_metriques_structure(qualifieur, jeu_lexique):
    echantillon = jeu_lexique.sample(200, random_state=0)
    r = ev.metriques(qualifieur, echantillon)
    assert 0 <= r["metier"]["f1_macro"] <= 1
    assert 0 <= r["metier"]["ece"] <= 1
    assert len(r["metier"]["matrice_confusion"]) == 7
    assert 0 <= r["revue_humaine"]["taux"] <= 1


def test_erreur_calibration_parfaite():
    probas = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert ev.erreur_calibration(probas, np.array(["a", "b"]), ["a", "b"]) == 0.0


def test_robustesse_toutes_perturbations(qualifieur):
    jeu = pd.DataFrame({"texte": ["Le robinet de la cuisine goutte"], "metier": ["plomberie"], "urgence": ["normale"]})
    r = ev.robustesse(qualifieur, jeu)
    assert set(r["perturbation"]) == set(ev.PERTURBATIONS)


def resultats(f1_metier=0.85, f1_urgence=0.7, rappel=0.9):
    bloc = {"metier": {"f1_macro": f1_metier}, "urgence": {"f1_macro": f1_urgence, "rappel_critique": rappel}}
    return {"hors_gabarits": bloc, "test_temporel": bloc}


def test_promotion_sans_champion():
    assert decider_promotion(resultats(), None) == (True, [])


def test_refus_plancher_urgence_critique():
    promu, raisons = decider_promotion(resultats(rappel=0.5), None)
    assert not promu and "rappel" in raisons[0]


def test_refus_si_regression_face_au_champion():
    promu, raisons = decider_promotion(resultats(f1_metier=0.80), resultats(f1_metier=0.86))
    assert not promu and any("régression metier" in r for r in raisons)


def test_tolerance_petite_baisse():
    assert decider_promotion(resultats(f1_metier=0.855), resultats(f1_metier=0.86))[0]
