"""Fixtures : un petit modèle TF-IDF entraîné à la volée (rapide, sans téléchargement)."""

import hashlib

import pytest
from sklearn.base import clone

from artelio_ia import augmentation
from artelio_ia.modele import Qualifieur, candidats

CLE_API = "cle-de-test"


@pytest.fixture(scope="session")
def jeu_lexique():
    return augmentation.generer(n_par_combinaison=40)


@pytest.fixture(scope="session")
def qualifieur(jeu_lexique):
    modele = candidats(avec_plongements=False)["regression_logistique_C4.0"]
    m = clone(modele).fit(jeu_lexique["texte"], jeu_lexique["metier"])
    u = clone(modele).fit(jeu_lexique["texte"], jeu_lexique["urgence"])
    return Qualifieur(m, u, {"entraine_le": "2026-10-26T00:00:00+00:00", "version_donnees": "test", "modeles": {}})


@pytest.fixture
def empreinte_cle():
    return hashlib.sha256(CLE_API.encode()).hexdigest()
