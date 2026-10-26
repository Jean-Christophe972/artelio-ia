"""Plongements de phrases (sentence embeddings) comme variables du modèle.

Un modèle TF-IDF ne connaît que les mots vus à l'entraînement : « ma fille attend dedans seule »
ne ressemble à aucun exemple de « enfant enfermé ». Un encodeur pré-entraîné sur des millions de
paires de paraphrases (MiniLM multilingue, 384 dimensions, CPU) place deux phrases de même SENS
près l'une de l'autre, même sans mot commun. On le combine au TF-IDF (vocabulaire du métier).

Le transformeur respecte l'interface scikit-learn ; le réseau n'est pas sérialisé avec le modèle
(il est chargé depuis l'image Docker, version figée) : l'artefact reste léger.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

MODELE_PAR_DEFAUT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Cache partagé par le processus, ACTIVÉ UNIQUEMENT PENDANT L'ENTRAÎNEMENT (les clones de la validation
# croisée réutilisent les encodages). Désactivé en production : il grossirait sans limite.
_CACHE: dict[tuple[str, str], np.ndarray] = {}
_CACHE_ACTIF = False
_ENCODEURS: dict[str, object] = {}


def activer_cache(actif: bool = True) -> None:
    global _CACHE_ACTIF
    _CACHE_ACTIF = actif
    if not actif:
        _CACHE.clear()


class PlongementsPhrases(BaseEstimator, TransformerMixin):
    def __init__(self, nom_modele: str = MODELE_PAR_DEFAUT, taille_lot: int = 64):
        self.nom_modele = nom_modele
        self.taille_lot = taille_lot

    def fit(self, X, y=None):
        self.n_features_out_ = self._encodeur().get_sentence_embedding_dimension()
        return self

    def transform(self, X) -> np.ndarray:
        # Le réseau est sensible à la casse : un message tapé EN MAJUSCULES serait encodé autrement
        # (constaté : -15 points d'exactitude). On normalise comme pour le TF-IDF.
        textes = [" ".join(str(t).lower().split()) for t in X]
        if not _CACHE_ACTIF:
            return self._encodeur().encode(textes, batch_size=self.taille_lot, normalize_embeddings=True, show_progress_bar=False)
        manquants = [t for t in dict.fromkeys(textes) if (self.nom_modele, t) not in _CACHE]
        if manquants:
            vecteurs = self._encodeur().encode(manquants, batch_size=self.taille_lot, normalize_embeddings=True, show_progress_bar=False)
            _CACHE.update({(self.nom_modele, t): v for t, v in zip(manquants, vecteurs, strict=True)})
        return np.vstack([_CACHE[(self.nom_modele, t)] for t in textes])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"dim_{i}" for i in range(self.n_features_out_)], dtype=object)

    def _encodeur(self):
        # Le réseau n'est jamais sérialisé avec le modèle : seul son nom l'est (version figée par l'image)
        if self.nom_modele not in _ENCODEURS:
            from sentence_transformers import SentenceTransformer

            _ENCODEURS[self.nom_modele] = SentenceTransformer(self.nom_modele, device="cpu")
        return _ENCODEURS[self.nom_modele]
