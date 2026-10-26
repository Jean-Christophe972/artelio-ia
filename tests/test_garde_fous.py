"""Garde-fous : détectent les signaux de danger, sans fausse alerte sur les demandes courantes."""

import pytest

from artelio_ia.garde_fous import signaux_critiques


@pytest.mark.parametrize(
    ("texte", "signal"),
    [
        ("Odeur de gaz dans la cuisine", "gaz"),
        ("le détecteur de monoxyde sonne", "monoxyde"),
        ("La prise fait des étincelles", "electrique_brule"),
        ("ça sent le brûlé derrière le tableau", "electrique_brule"),
        ("L'eau jaillit du tuyau, tout est inondé", "inondation"),
        ("Je suis bloqué dehors, la porte a claqué", "bloque_dehors"),
        ("Mon bébé est enfermé à l'intérieur", "enfant_enferme"),
        ("On a été cambriolés cette nuit", "effraction"),
        ("Plus de chauffage et il gèle, ma mère est âgée", "froid_vulnerable"),
    ],
)
def test_signal_detecte(texte, signal):
    assert signal in signaux_critiques(texte)


@pytest.mark.parametrize(
    "texte",
    [
        "Repeindre le salon en blanc",
        "Le robinet goutte un peu",
        "Plafond taché après un dégât des eaux réparé",  # pas d'urgence : dégât ancien
        "Installer une prise dans la chambre de l'enfant",
        "Entretien annuel de la chaudière gaz",
    ],
)
def test_pas_de_fausse_alerte(texte):
    assert signaux_critiques(texte) == []
