"""Barrière de qualité du modèle livré (models/champion), exécutée par la CI avant tout déploiement.

Nécessite le modèle de plongements (image Docker ou téléchargement en CI) : marqué « champion ».
"""

import pytest

from artelio_ia import evaluation as ev
from artelio_ia import registre
from artelio_ia.config import DOSSIER_MODELES, MACRO_F1_METIER_MIN, RAPPEL_CRITIQUE_MIN
from artelio_ia.donnees import charger_hors_gabarits

pytestmark = pytest.mark.champion


@pytest.fixture(scope="module")
def champion():
    return registre.charger_local(DOSSIER_MODELES / "champion")


def test_planchers_sur_validation(champion):
    r = ev.metriques(champion, charger_hors_gabarits("validation"))
    assert r["metier"]["f1_macro"] >= MACRO_F1_METIER_MIN
    assert r["urgence"]["rappel_critique"] >= RAPPEL_CRITIQUE_MIN


def test_aucune_sous_estimation_grave_sur_test(champion):
    # Une urgence critique classée « faible » ou « normale » est l'erreur la plus coûteuse
    assert ev.metriques(champion, charger_hors_gabarits("test"))["urgence"]["sous_estimation_grave"] <= 0.02


# Perte d'exactitude métier maximale tolérée par perturbation (en points de pourcentage).
# Fautes de frappe denses (6 % des lettres inversées) : LIMITE CONNUE, mesurée à -11,2 points sur le champion v3.
# L'entraînement sur copies bruitées (v4) ne gagnait que 1,4 point et dégradait l'urgence : refusé par la
# barrière de promotion. Le test empêche toute aggravation au-delà de la limite documentée.
TOLERANCES = {"sans_accents": 0.10, "majuscules": 0.10, "telephone_ajoute": 0.10, "bruit_politesse": 0.10, "fautes_de_frappe": 0.12}


def test_robustesse(champion):
    r = ev.robustesse(champion, charger_hors_gabarits("validation")).set_index("perturbation")
    reference = r.loc["original", "exactitude_metier"]
    for perturbation, tolerance in TOLERANCES.items():
        assert reference - r.loc[perturbation, "exactitude_metier"] <= tolerance, perturbation
