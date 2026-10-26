"""Masquage des données personnelles dans les textes libres (repris du pipeline, bloc 3).

Un client écrit parfois son numéro ou son email dans la description de sa demande.
Ces informations ne doivent pas se retrouver dans les jeux d'entraînement de l'IA ni
dans la file d'erreurs : elles sont remplacées par des marqueurs neutres.
"""

from __future__ import annotations

import re

_MOTIFS: list[tuple[str, re.Pattern[str]]] = [
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE)),
    # IBAN français (FR76 suivi de 23 caractères, avec ou sans espaces)
    ("[IBAN]", re.compile(r"\bFR\d{2}(?:\s?[0-9A-Z]{4}){5}\s?[0-9A-Z]{3}\b", re.IGNORECASE)),
    # Téléphones français : 06 12 34 56 78, 06.12.34.56.78, 0612345678, +33 6 12 34 56 78, 0033612345678
    ("[TELEPHONE]", re.compile(r"(?<!\d)(?:(?:\+|00)33\s?[1-9]|0[1-9])(?:[\s.-]?\d{2}){4}(?!\d)")),
]


def masquer(texte: str) -> tuple[str, int]:
    """Renvoie le texte masqué et le nombre de remplacements effectués."""
    total = 0
    for marqueur, motif in _MOTIFS:
        texte, n = motif.subn(marqueur, texte)
        total += n
    return texte, total


def contient_donnees_personnelles(texte: str) -> bool:
    """Contrôle de sortie : vrai si un motif de donnée personnelle subsiste."""
    return any(motif.search(texte) for _, motif in _MOTIFS)
