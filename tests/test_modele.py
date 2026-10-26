"""Le modèle et sa politique de décision (revue humaine, garde-fous, explication)."""

from artelio_ia.config import METIERS, URGENCES


def test_prediction_complete(qualifieur):
    p = qualifieur.predire(["Le robinet de la cuisine goutte sans arrêt"], expliquer=True)[0]
    assert p.metier == "plomberie"
    assert p.metier in METIERS and p.urgence in URGENCES
    assert abs(sum(p.probas_metier.values()) - 1) < 1e-3
    assert any(t["terme"] == "robinet" for t in p.explication)


def test_garde_fou_force_critique(qualifieur):
    texte = "Ça sent le gaz près de la chaudière"
    avec = qualifieur.predire([texte])[0]
    assert avec.urgence == "critique"
    qualifieur.garde_fous = False
    try:
        sans = qualifieur.predire([texte])[0]
    finally:
        qualifieur.garde_fous = True
    # Le garde-fou n'intervient que si le modèle ne voyait pas déjà « critique »
    assert sans.urgence == "critique" or any(m.startswith("garde_fou:gaz") for m in avec.motifs_revue)


def test_revue_humaine_si_texte_hors_domaine(qualifieur):
    p = qualifieur.predire(["Quel temps fera-t-il demain à Lyon ?"])[0]
    assert p.revue_humaine
    assert {"confiance_metier_faible", "hors_domaine"} & set(p.motifs_revue)


def test_deterministe(qualifieur):
    a = qualifieur.predire(["serrure bloquée"])[0]
    b = qualifieur.predire(["serrure bloquée"])[0]
    assert a.probas_metier == b.probas_metier
