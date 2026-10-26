"""Sérialisation : un artefact modifié est refusé au chargement."""

import pytest

from artelio_ia import registre


def test_aller_retour(qualifieur, tmp_path):
    registre.sauvegarder(qualifieur, tmp_path)
    recharge = registre.charger_local(tmp_path)
    texte = ["serrure bloquée, impossible de fermer"]
    assert recharge.predire(texte)[0].probas_metier == qualifieur.predire(texte)[0].probas_metier


def test_artefact_altere_refuse(qualifieur, tmp_path):
    chemin = registre.sauvegarder(qualifieur, tmp_path)
    with open(chemin, "ab") as f:
        f.write(b"code malveillant")
    with pytest.raises(registre.ArtefactCorrompu):
        registre.charger_local(tmp_path)
