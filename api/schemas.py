"""Contrats d'entrée et de sortie de l'API (validés et documentés automatiquement dans /docs)."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DemandeAQualifier(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"texte": "Fuite d'eau sous l'évier, ça coule partout dans la cuisine", "canal": "app"}},
    )

    texte: str = Field(min_length=3, max_length=2000, description="Description du problème, telle qu'écrite par le client")
    canal: Literal["app", "web", "vocal", "telephone"] = "app"
    demande_id: uuid.UUID | None = Field(default=None, description="Identifiant de la demande (traçabilité)")
    expliquer: bool = Field(default=True, description="Renvoyer les termes qui ont pesé dans la décision")

    @field_validator("texte")
    @classmethod
    def _nettoyer(cls, valeur: str) -> str:
        valeur = "".join(c if c.isprintable() else " " for c in valeur).strip()
        if len(valeur) < 3:
            raise ValueError("texte vide après nettoyage")
        return valeur


class LotAQualifier(BaseModel):
    model_config = ConfigDict(extra="forbid")
    demandes: list[DemandeAQualifier] = Field(min_length=1, max_length=100)


class Terme(BaseModel):
    terme: str
    contribution: float | None


class Qualification(BaseModel):
    demande_id: uuid.UUID | None
    metier: str
    confiance_metier: float = Field(description="Probabilité du métier prédit")
    urgence: Literal["faible", "normale", "haute", "critique"]
    confiance_urgence: float
    probabilites_metier: dict[str, float]
    probabilites_urgence: dict[str, float]
    revue_humaine: bool = Field(description="Vrai si un humain doit valider (confiance faible ou urgence critique suspectée)")
    motifs_revue: list[str]
    explication: list[Terme]
    version_modele: str
    avertissement: str = "Pré-qualification automatique par IA ; un conseiller peut la corriger."


class Lot(BaseModel):
    resultats: list[Qualification]


class InfoModele(BaseModel):
    version_modele: str
    entraine_le: str | None
    version_donnees: str | None
    modeles: dict | None
    seuils: dict | None
    metriques: dict | None
