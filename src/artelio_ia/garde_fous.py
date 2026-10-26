"""Garde-fous déterministes pour les urgences critiques.

Les motifs décrivent un vocabulaire de DANGER générique (gaz, fumée, étincelles, inondation,
effraction, personne vulnérable) ; ils ne reprennent pas les phrases des jeux de test.

Le coût des erreurs est asymétrique : rater une fuite de gaz ou un enfant enfermé est grave,
envoyer un artisan en urgence pour rien l'est beaucoup moins. En plus du modèle, des règles
explicites, relues par le métier, forcent l'urgence "critique" quand un signal de danger est
présent. Elles sont lisibles, testées, et leur effet est mesuré séparément (rapport d'évaluation).
"""

from __future__ import annotations

import re
import unicodedata


def _normaliser(texte: str) -> str:
    texte = "".join(c for c in unicodedata.normalize("NFD", texte.lower()) if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", texte)


# (nom du signal, motif sur le texte sans accents et en minuscules)
SIGNAUX_CRITIQUES: list[tuple[str, re.Pattern[str]]] = [
    ("gaz", re.compile(r"\b(odeur de gaz|fuite de gaz|sent le gaz|ca sent le gaz)\b")),
    ("monoxyde", re.compile(r"\b(monoxyde|detecteur de co)\b")),
    ("feu_fumee", re.compile(r"\b(fumee|pris feu|flammes?|incendie)\b")),
    ("electrique_brule", re.compile(r"\b(etincelles?|crame|sent le brule|odeur de brule|fils? (electriques? )?denudes?)\b")),
    ("inondation", re.compile(r"\b(inonde|inondation|jaillit|eclate|deborde|chez le voisin|plafond du voisin)\b")),
    ("bloque_dehors", re.compile(r"\b(bloquee? dehors|enfermee? dehors|porte (a )?claque|impossible de rentrer)\b")),
    ("enfant_enferme", re.compile(r"\b(bebe|nourrisson|enfant)\b.{0,60}\b(enferme|a l'interieur|dedans|bloque)")),
    ("effraction", re.compile(r"\b(cambriol\w*|fractur\w*|forcee|effraction)\b")),
    (
        "froid_vulnerable",
        re.compile(
            r"\b(gele|il gele|grand froid)\b.{0,80}\b(bebe|nourrisson|agee|malade)\b"
            r"|\b(bebe|nourrisson|agee|malade)\b.{0,80}\b(gele|grand froid|chauffage (est )?coupe|plus de chauffage)\b"
        ),
    ),
]


def signaux_critiques(texte: str) -> list[str]:
    """Noms des signaux de danger détectés dans le texte."""
    t = _normaliser(texte)
    return [nom for nom, motif in SIGNAUX_CRITIQUES if motif.search(t)]
