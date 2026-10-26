"""Le modèle de qualification des demandes : métier (7 classes) et urgence (4 classes).

Choix : TF-IDF (mots 1-2 + caractères 2-5) + régression logistique, un classifieur par tâche.
  - Les n-grammes de CARACTÈRES rendent le modèle robuste aux fautes de frappe et aux
    abréviations (« apart », « clé cassé », « chaudiere »), fréquentes sur mobile.
  - La normalisation des accents (strip_accents) traite « chaudière » et « chaudiere » pareil.
  - La régression logistique donne des PROBABILITÉS (nécessaires à la revue humaine) et des
    poids lisibles (explicabilité, AI Act) ; elle prédit en quelques millisecondes sur CPU.
  - class_weight="balanced" compense les métiers rares (menuiserie, petits travaux).
  - Des garde-fous déterministes (garde_fous.py) forcent l'urgence critique sur les signaux de danger.
Alternative écartée pour l'instant : CamemBERT (meilleure compréhension du sens, mais beaucoup
plus lent, besoin de GPU, explicabilité plus difficile) — voir docs/MODEL_CARD.md.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from artelio_ia.config import GRAINE, SEUIL_ALERTE_CRITIQUE, SEUIL_CONFIANCE_METIER, SEUIL_HORS_DOMAINE, URGENCES
from artelio_ia.garde_fous import signaux_critiques
from artelio_ia.plongements import PlongementsPhrases


def normaliser(texte: str) -> str:
    """Nettoyage léger : minuscules, espaces, caractères de contrôle. Les accents sont gérés par le TF-IDF."""
    texte = unicodedata.normalize("NFKC", texte).lower()
    texte = re.sub(r"[\x00-\x1f\x7f]", " ", texte)
    return re.sub(r"\s+", " ", texte).strip()


def vectoriseur() -> FeatureUnion:
    return FeatureUnion(
        [
            ("mots", TfidfVectorizer(preprocessor=normaliser, strip_accents="unicode", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            (
                "caracteres",
                TfidfVectorizer(
                    preprocessor=normaliser, strip_accents="unicode", analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
                ),
            ),
        ]
    )


def vectoriseur_hybride(poids_plongements: float = 1.5) -> FeatureUnion:
    """TF-IDF (vocabulaire du métier, robustesse aux fautes) + plongements (sens de la phrase)."""
    union = vectoriseur()
    union.transformer_list.append(("plongements", PlongementsPhrases()))
    union.transformer_weights = {"mots": 1.0, "caracteres": 1.0, "plongements": poids_plongements}
    return union


def candidats(avec_plongements: bool = True) -> dict[str, Pipeline]:
    """Modèles comparés en validation croisée (voir entrainement.py et le notebook)."""
    return {
        "reference_classe_majoritaire": Pipeline([("vect", vectoriseur()), ("clf", DummyClassifier(strategy="most_frequent"))]),
        "naive_bayes": Pipeline([("vect", vectoriseur()), ("clf", ComplementNB(alpha=0.3))]),
        "svm_lineaire": Pipeline([("vect", vectoriseur()), ("clf", LinearSVC(C=0.5, class_weight="balanced", random_state=GRAINE))]),
        **{
            f"regression_logistique_C{c}": Pipeline(
                [("vect", vectoriseur()), ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=GRAINE))]
            )
            for c in (1.0, 4.0, 16.0)
        },
        **(
            {
                "plongements_regression_logistique": Pipeline(
                    [
                        ("vect", PlongementsPhrases()),
                        ("clf", LogisticRegression(C=4.0, class_weight="balanced", max_iter=3000, random_state=GRAINE)),
                    ]
                ),
                **{
                    f"hybride_regression_logistique_C{c}": Pipeline(
                        [
                            ("vect", vectoriseur_hybride()),
                            ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=GRAINE)),
                        ]
                    )
                    for c in (1.0, 4.0)
                },
            }
            if avec_plongements
            else {}
        ),
    }


@dataclass
class Prediction:
    metier: str
    proba_metier: float
    urgence: str
    proba_urgence: float
    probas_metier: dict[str, float]
    probas_urgence: dict[str, float]
    revue_humaine: bool
    motifs_revue: list[str]
    explication: list[dict] = field(default_factory=list)


@dataclass
class Qualifieur:
    """Les deux modèles + la politique de décision, sérialisés ensemble (un seul artefact versionné)."""

    modele_metier: Pipeline
    modele_urgence: Pipeline
    metadonnees: dict = field(default_factory=dict)
    garde_fous: bool = True  # désactivable pour mesurer l'apport des règles (ablation)

    def predire(self, textes: list[str], expliquer: bool = False) -> list[Prediction]:
        p_metier = self.modele_metier.predict_proba(textes)
        p_urgence = self.modele_urgence.predict_proba(textes)
        classes_m = list(self.modele_metier.classes_)
        classes_u = list(self.modele_urgence.classes_)
        resultats = []
        for i, texte in enumerate(textes):
            im = int(np.argmax(p_metier[i]))
            iu = int(np.argmax(p_urgence[i]))
            urgence = classes_u[iu]
            motifs = []
            if p_metier[i][im] < SEUIL_CONFIANCE_METIER:
                motifs.append("confiance_metier_faible")
            if self.taux_hors_vocabulaire(texte) >= SEUIL_HORS_DOMAINE:
                motifs.append("hors_domaine")
            p_critique = float(p_urgence[i][classes_u.index("critique")])
            if urgence != "critique" and p_critique >= SEUIL_ALERTE_CRITIQUE:
                urgence = "critique"  # principe de précaution
                motifs.append("suspicion_urgence_critique")
            if self.garde_fous and urgence != "critique" and (signaux := signaux_critiques(texte)):
                urgence = "critique"  # signal de danger explicite : la règle l'emporte sur le modèle
                motifs.append("garde_fou:" + ",".join(signaux))
            resultats.append(
                Prediction(
                    metier=classes_m[im],
                    proba_metier=round(float(p_metier[i][im]), 4),
                    urgence=urgence,
                    proba_urgence=round(float(p_urgence[i][classes_u.index(urgence)]), 4),
                    probas_metier={c: round(float(p), 4) for c, p in zip(classes_m, p_metier[i], strict=True)},
                    probas_urgence={c: round(float(p), 4) for c, p in zip(classes_u, p_urgence[i], strict=True)},
                    revue_humaine=bool(motifs),
                    motifs_revue=motifs,
                    explication=self.expliquer(texte, classes_m[im]) if expliquer else [],
                )
            )
        return resultats

    def expliquer(self, texte: str, classe: str, n: int = 6) -> list[dict]:
        """Les termes du texte qui ont le plus pesé dans le choix du métier (poids × TF-IDF)."""
        vect = self.modele_metier.named_steps["vect"]
        clf = self.modele_metier.named_steps["clf"]
        if not hasattr(vect, "transformer_list"):  # modèle à plongements seuls : pas de termes lisibles
            return [{"terme": "(sens global de la phrase)", "contribution": None}]
        from scipy import sparse

        x = sparse.csr_matrix(vect.transform([texte]))
        coefs = clf.coef_[list(clf.classes_).index(classe)]
        noms = vect.get_feature_names_out()
        contributions, sens_global = [], 0.0
        x = x.tocsr() if hasattr(x, "tocsr") else x
        for j, valeur in zip(x.indices, x.data, strict=True):
            nom = str(noms[j])
            if nom.startswith("mots__"):  # les n-grammes de mots sont lisibles par un humain
                contributions.append((nom.removeprefix("mots__"), float(valeur * coefs[j])))
            elif nom.startswith("plongements__"):
                sens_global += float(valeur * coefs[j])
        if sens_global:
            contributions.append(("(sens global de la phrase)", sens_global))
        contributions.sort(key=lambda c: c[1], reverse=True)
        return [{"terme": t, "contribution": round(c, 4)} for t, c in contributions[:n] if c > 0]

    def taux_hors_vocabulaire(self, texte: str) -> float:
        """Part des mots du texte inconnus du vocabulaire d'entraînement (indicateur de hors domaine et de dérive)."""
        vect = self.modele_metier.named_steps["vect"]
        mots_vect = dict(vect.transformer_list).get("mots") if hasattr(vect, "transformer_list") else None
        if mots_vect is None:
            return 0.0
        mots = [m for m in mots_vect.build_analyzer()(texte) if " " not in m]
        return sum(m not in mots_vect.vocabulary_ for m in mots) / len(mots) if mots else 0.0

    def latence_ms(self, repetitions: int = 100) -> float:
        """Latence d'une prédiction unitaire, sur des textes tous différents (aucun effet de cache)."""
        textes = [f"fuite d'eau sous l'évier numéro {i}, ça coule depuis {i} minutes" for i in range(repetitions)]
        debut = time.perf_counter()
        for texte in textes:
            self.predire([texte])
        return 1000 * (time.perf_counter() - debut) / repetitions


def ordre_urgence(urgence: str) -> int:
    return URGENCES.index(urgence)
