"""Chargement des données d'entraînement et découpage train / test.

Le jeu d'entraînement est publié par le pipeline (bloc 3) dans S3, versionné, avec un
manifeste contenant son empreinte SHA-256. On vérifie l'empreinte avant tout usage :
le modèle est toujours entraîné sur une version identifiée et intacte (reproductibilité).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from dataclasses import dataclass

import boto3
import pandas as pd
import psycopg
from botocore.config import Config as BotoConfig

from artelio_ia.config import DOSSIER_DATA, JOURS_TEST_TEMPOREL, METIERS, URGENCES, env

log = logging.getLogger(__name__)
BUCKET = "datasets"


class DonneesInvalides(Exception):
    """Le jeu de données est corrompu ou ne respecte pas le schéma attendu."""


@dataclass
class JeuDeDonnees:
    version: str
    sha256: str
    donnees: pd.DataFrame
    manifest: dict


def _s3():
    return boto3.client(
        "s3",
        endpoint_url=env("S3_ENDPOINT", "http://object-storage:8333"),
        aws_access_key_id=env("S3_ACCESS_KEY"),
        aws_secret_access_key=env("S3_SECRET_KEY"),
        region_name="eu-west-3",
        config=BotoConfig(retries={"max_attempts": 5, "mode": "standard"}, s3={"addressing_style": "path"}),
    )


def derniere_version() -> str:
    corps = _s3().get_object(Bucket=BUCKET, Key="demandes/latest.json")["Body"].read()
    return json.loads(corps)["version"]


def charger(version: str | None = None) -> JeuDeDonnees:
    """Télécharge une version du jeu (la dernière par défaut) et vérifie son intégrité."""
    s3 = _s3()
    version = version or derniere_version()
    manifest = json.loads(s3.get_object(Bucket=BUCKET, Key=f"demandes/{version}/manifest.json")["Body"].read())
    contenu = s3.get_object(Bucket=BUCKET, Key=manifest["fichier"])["Body"].read()
    empreinte = hashlib.sha256(contenu).hexdigest()
    reference = empreinte_de_reference(version) or manifest["sha256"]
    if empreinte != reference:
        raise DonneesInvalides(f"empreinte SHA-256 du jeu {version} différente de la référence : jeu modifié, entraînement refusé")
    donnees = pd.read_csv(io.BytesIO(contenu), parse_dates=["cree_le"])
    valider_schema(donnees)
    log.info("Jeu %s chargé : %d lignes, intégrité vérifiée (sha256 %s…)", version, len(donnees), empreinte[:12])
    return JeuDeDonnees(version, empreinte, donnees, manifest)


def empreinte_de_reference(version: str) -> str | None:
    """Empreinte enregistrée par le pipeline dans PostgreSQL (pipeline.datasets).

    C'est l'ancre d'intégrité : le service IA peut écrire dans le bucket S3 (contrainte technique
    de SeaweedFS), mais pas dans ce catalogue. Un jeu modifié dans S3 serait donc détecté.
    """
    if not os.environ.get("PG_HOST"):
        return None
    with psycopg.connect(
        host=env("PG_HOST"),
        dbname=env("PG_DATABASE", "artelio"),
        user=env("PG_USER", "ml_service"),
        password=env("PG_PASSWORD"),
        sslmode="require",
        connect_timeout=5,
    ) as conn:
        row = conn.execute("SELECT sha256 FROM pipeline.datasets WHERE version = %s", (version,)).fetchone()
    if row is None:
        raise DonneesInvalides(f"version {version} absente du catalogue pipeline.datasets")
    return row[0]


def valider_schema(donnees: pd.DataFrame) -> None:
    """Contrat minimal sur les données d'entrée du modèle."""
    manquantes = {"texte", "metier", "urgence", "cree_le"} - set(donnees.columns)
    if manquantes:
        raise DonneesInvalides(f"colonnes manquantes : {sorted(manquantes)}")
    if donnees["texte"].isna().any() or (donnees["texte"].str.len() == 0).any():
        raise DonneesInvalides("textes vides")
    if inconnus := set(donnees["metier"]) - set(METIERS):
        raise DonneesInvalides(f"métiers inconnus : {sorted(inconnus)}")
    if inconnues := set(donnees["urgence"]) - set(URGENCES):
        raise DonneesInvalides(f"urgences inconnues : {sorted(inconnues)}")


def decouper_temporel(donnees: pd.DataFrame, jours_test: int = JOURS_TEST_TEMPOREL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Les derniers jours servent de test : on évalue sur le futur, jamais sur le passé mélangé.

    Un découpage aléatoire surestimerait la performance (fuite temporelle, saisonnalité).
    """
    coupure = donnees["cree_le"].max() - pd.Timedelta(days=jours_test)
    train = donnees[donnees["cree_le"] <= coupure].reset_index(drop=True)
    test = donnees[donnees["cree_le"] > coupure].reset_index(drop=True)
    return train, test


def charger_hors_gabarits(nom: str = "validation") -> pd.DataFrame:
    """Jeux écrits à la main, formulés différemment des données synthétiques (voir data/README.md).

    - "validation" : sert au développement (diagnostic, choix des améliorations) ;
    - "test" : rédigé APRÈS le gel du modèle v2, jamais utilisé pour régler quoi que ce soit.
    """
    donnees = pd.read_csv(DOSSIER_DATA / f"{nom}_hors_gabarits.csv")
    donnees["cree_le"] = pd.Timestamp("2026-10-01")
    valider_schema(donnees)
    return donnees
