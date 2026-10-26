"""Contrat des données, découpage temporel, enrichissement, masquage."""

import pandas as pd
import pytest

from artelio_ia import augmentation
from artelio_ia.donnees import DonneesInvalides, charger_hors_gabarits, decouper_temporel, valider_schema
from artelio_ia.pii import masquer


def test_decoupage_temporel_sans_fuite():
    jeu = pd.DataFrame(
        {"cree_le": pd.date_range("2026-01-01", periods=200, freq="D"), "texte": "x", "metier": "plomberie", "urgence": "faible"}
    )
    train, test = decouper_temporel(jeu, jours_test=60)
    assert train["cree_le"].max() < test["cree_le"].min()  # aucune donnée du futur à l'entraînement
    assert len(train) + len(test) == len(jeu)


def test_schema_refuse_metier_inconnu():
    jeu = pd.DataFrame({"texte": ["a"], "metier": ["jardinage"], "urgence": ["faible"], "cree_le": [pd.Timestamp("2026-01-01")]})
    with pytest.raises(DonneesInvalides, match="métiers inconnus"):
        valider_schema(jeu)


@pytest.mark.parametrize("nom", ["validation", "test"])
def test_jeux_hors_gabarits_valides(nom):
    jeu = charger_hors_gabarits(nom)
    assert len(jeu) >= 90
    assert set(jeu["urgence"]) == {"faible", "normale", "haute", "critique"}


def test_jeux_de_validation_et_test_disjoints():
    a, b = set(charger_hors_gabarits("validation")["texte"]), set(charger_hors_gabarits("test")["texte"])
    assert not a & b


def test_enrichissement_equilibre_et_deterministe():
    a, b = augmentation.generer(10), augmentation.generer(10)
    assert a.equals(b)  # même graine → mêmes exemples (reproductibilité)
    assert a.groupby(["metier", "urgence"]).size().nunique() == 1


def test_masquage_avant_journalisation():
    texte, n = masquer("rappelez-moi au 06 12 34 56 78 ou jean@example.org")
    assert n == 2 and "06" not in texte and "@" not in texte
