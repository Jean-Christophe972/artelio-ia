"""Sérialisation du modèle et registre MLflow (versions, alias « champion »).

Le modèle est enregistré comme un modèle MLflow "pyfunc" contenant l'artefact joblib.
Sécurité : un fichier joblib (pickle) peut exécuter du code au chargement ; on ne charge
donc que des artefacts de notre registre, après vérification de leur empreinte SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import joblib

from artelio_ia.config import NOM_MODELE
from artelio_ia.modele import Qualifieur

log = logging.getLogger(__name__)
FICHIER = "qualifieur.joblib"


class ArtefactCorrompu(Exception):
    """L'empreinte de l'artefact ne correspond pas à celle enregistrée : chargement refusé."""


def sha256_fichier(chemin: Path) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def sauvegarder(qualifieur: Qualifieur, dossier: Path) -> Path:
    """Écrit le modèle et ses métadonnées (dont l'empreinte) dans un dossier."""
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / FICHIER
    joblib.dump(qualifieur, chemin, compress=3)
    meta = dict(qualifieur.metadonnees, sha256_artefact=sha256_fichier(chemin))
    (dossier / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return chemin


def charger_local(dossier: Path) -> Qualifieur:
    """Charge un modèle exporté (ex. models/champion), après vérification de l'empreinte."""
    meta = json.loads((dossier / "metadata.json").read_text(encoding="utf-8"))
    chemin = dossier / FICHIER
    if sha256_fichier(chemin) != meta["sha256_artefact"]:
        raise ArtefactCorrompu(f"empreinte invalide pour {chemin}")
    qualifieur = joblib.load(chemin)
    qualifieur.metadonnees = meta
    return qualifieur


def charger_depuis_registre(alias: str = "champion") -> Qualifieur | None:
    """Télécharge la version du registre portant l'alias, vérifie son empreinte et la charge."""
    import mlflow
    from mlflow import MlflowClient

    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(NOM_MODELE, alias)
    except mlflow.exceptions.MlflowException:
        return None
    dossier = Path(mlflow.artifacts.download_artifacts(f"models:/{NOM_MODELE}@{alias}"))
    chemin = next(dossier.rglob(FICHIER))
    attendu = version.tags.get("sha256_artefact")
    if attendu and sha256_fichier(chemin) != attendu:
        raise ArtefactCorrompu(f"empreinte invalide pour la version {version.version}")
    qualifieur = joblib.load(chemin)
    qualifieur.metadonnees = dict(qualifieur.metadonnees, version_registre=version.version, alias=alias)
    return qualifieur


def version_alias(alias: str = "champion") -> str | None:
    """Numéro de version portant l'alias (utilisé par l'API pour le rechargement à chaud)."""
    import mlflow
    from mlflow import MlflowClient

    try:
        return MlflowClient().get_model_version_by_alias(NOM_MODELE, alias).version
    except mlflow.exceptions.MlflowException:
        return None


try:  # le wrapper pyfunc n'est utile que là où MLflow complet est installé (entraînement)
    import mlflow.pyfunc

    class ModeleArtelio(mlflow.pyfunc.PythonModel):
        """Enveloppe MLflow : permet de servir le modèle avec les outils standard de MLflow."""

        def load_context(self, context):
            self.qualifieur = joblib.load(context.artifacts["qualifieur"])

        def predict(self, context, model_input, params=None):
            import pandas as pd

            preds = self.qualifieur.predire(model_input["texte"].tolist())
            return pd.DataFrame([{"metier": p.metier, "urgence": p.urgence, "confiance": p.proba_metier} for p in preds])

except ImportError:  # pragma: no cover — image API (mlflow-skinny)
    ModeleArtelio = None
