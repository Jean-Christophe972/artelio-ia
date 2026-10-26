"""API de qualification des demandes Artelio (FastAPI).

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Documentation interactive : /docs. Sécurité : clé d'API (en-tête X-API-Key, seule son empreinte
SHA-256 est connue du serveur), limitation de débit, validation stricte des entrées, aucune
trace d'erreur interne renvoyée au client, texte masqué avant journalisation.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from api.schemas import DemandeAQualifier, InfoModele, Lot, LotAQualifier, Qualification
from api.service import JournalPredictions, ModeleServi, ligne_journal, taux_hors_vocabulaire

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s — %(message)s")
log = logging.getLogger("api")

# --- Métriques Prometheus -------------------------------------------------------------------
REQUETES = Counter("artelio_ia_requetes_total", "Requêtes HTTP", ["route", "code"])
LATENCE = Histogram(
    "artelio_ia_latence_secondes", "Durée de traitement d'une requête", ["route"], buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5)
)
PREDICTIONS = Counter("artelio_ia_predictions_total", "Prédictions", ["metier", "urgence"])
REVUE = Counter("artelio_ia_revue_humaine_total", "Prédictions envoyées en revue humaine", ["motif"])
CONFIANCE = Histogram(
    "artelio_ia_confiance_metier", "Confiance du modèle sur le métier", buckets=(0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1)
)
HORS_VOCAB = Histogram(
    "artelio_ia_taux_hors_vocabulaire",
    "Part de mots inconnus du vocabulaire d'entraînement",
    buckets=(0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1),
)
MODELE_INFO = Gauge("artelio_ia_modele_info", "Version du modèle servi", ["version"])

modele = ModeleServi()
journal = JournalPredictions()


@asynccontextmanager
async def cycle_de_vie(_app: FastAPI):
    modele.charger()
    MODELE_INFO.labels(modele.version).set(1)
    modele.surveiller_registre()
    yield


app = FastAPI(
    title="Artelio — API de qualification des demandes",
    version="1.0.0",
    description="Prédit le métier et l'urgence d'une demande de dépannage, avec explication et revue humaine si nécessaire.",
    lifespan=cycle_de_vie,
)


# --- Sécurité -------------------------------------------------------------------------------
cle_api = APIKeyHeader(name="X-API-Key", auto_error=False)
EMPREINTES = [e.strip() for e in os.environ.get("API_KEY_SHA256", "").split(",") if e.strip()]


class LimiteurDebit:
    """Seau à jetons par clé : 20 requêtes/s en continu, rafales de 40 tolérées."""

    def __init__(self, debit: float = 20.0, capacite: int = 40):
        self.debit, self.capacite, self.seaux, self.verrou = debit, capacite, {}, threading.Lock()

    def autoriser(self, cle: str) -> bool:
        maintenant = time.monotonic()
        with self.verrou:
            jetons, dernier = self.seaux.get(cle, (self.capacite, maintenant))
            jetons = min(self.capacite, jetons + (maintenant - dernier) * self.debit)
            if jetons < 1:
                self.seaux[cle] = (jetons, maintenant)
                return False
            self.seaux[cle] = (jetons - 1, maintenant)
            return True


limiteur = LimiteurDebit()


def authentifier(cle: str | None = Security(cle_api)) -> str:
    if not cle or not EMPREINTES:  # sans clé configurée, l'API refuse tout (échec sûr)
        raise HTTPException(status_code=401, detail="clé d'API manquante ou invalide")
    empreinte = hashlib.sha256(cle.encode()).hexdigest()
    if not any(hmac.compare_digest(empreinte, e) for e in EMPREINTES):
        raise HTTPException(status_code=401, detail="clé d'API manquante ou invalide")
    if not limiteur.autoriser(empreinte):
        raise HTTPException(status_code=429, detail="trop de requêtes, réessayez dans un instant")
    return empreinte


@app.middleware("http")
async def mesurer(request: Request, call_next):
    debut = time.perf_counter()
    route = request.scope.get("route").path if request.scope.get("route") else request.url.path
    try:
        reponse = await call_next(request)
    except Exception:  # noqa: BLE001 — jamais de trace interne renvoyée au client
        log.exception("Erreur interne sur %s", request.url.path)
        reponse = JSONResponse(status_code=500, content={"detail": "erreur interne"})
    route = request.scope.get("route").path if request.scope.get("route") else route
    REQUETES.labels(route, str(reponse.status_code)).inc()
    LATENCE.labels(route).observe(time.perf_counter() - debut)
    reponse.headers["X-Content-Type-Options"] = "nosniff"
    reponse.headers["Cache-Control"] = "no-store"
    return reponse


# --- Routes ---------------------------------------------------------------------------------
@app.get("/sante", tags=["exploitation"])
def sante():
    """Vivacité : le processus répond."""
    return {"statut": "ok"}


@app.get("/pret", tags=["exploitation"])
def pret():
    """Disponibilité : le modèle est chargé (utilisé par Kubernetes / Railway avant d'envoyer du trafic)."""
    if modele.qualifieur is None:
        raise HTTPException(status_code=503, detail="modèle non chargé")
    return {"statut": "pret", "version_modele": modele.version}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/modele", response_model=InfoModele, tags=["modèle"])
def info_modele(_cle: str = Depends(authentifier)):
    meta = modele.qualifieur.metadonnees
    return InfoModele(
        version_modele=modele.version,
        entraine_le=meta.get("entraine_le"),
        version_donnees=meta.get("version_donnees"),
        modeles=meta.get("modeles"),
        seuils=meta.get("seuils"),
        metriques=meta.get("metriques"),
    )


def _qualifier(demandes: list[DemandeAQualifier]) -> list[Qualification]:
    qualifieur, version = modele.qualifieur, modele.version
    debut = time.perf_counter()
    predictions = qualifieur.predire([d.texte for d in demandes], expliquer=any(d.expliquer for d in demandes))
    latence_ms = 1000 * (time.perf_counter() - debut) / len(demandes)
    resultats = []
    for demande, p in zip(demandes, predictions, strict=True):
        hors_vocab = taux_hors_vocabulaire(qualifieur, demande.texte)
        PREDICTIONS.labels(p.metier, p.urgence).inc()
        CONFIANCE.observe(p.proba_metier)
        HORS_VOCAB.observe(hors_vocab)
        for motif in p.motifs_revue:
            REVUE.labels(motif.split(":")[0]).inc()
        journal.ajouter(ligne_journal(demande.texte, demande.canal, demande.demande_id, p, version, hors_vocab, latence_ms))
        resultats.append(
            Qualification(
                demande_id=demande.demande_id,
                metier=p.metier,
                confiance_metier=p.proba_metier,
                urgence=p.urgence,
                confiance_urgence=p.proba_urgence,
                probabilites_metier=p.probas_metier,
                probabilites_urgence=p.probas_urgence,
                revue_humaine=p.revue_humaine,
                motifs_revue=p.motifs_revue,
                explication=p.explication if demande.expliquer else [],
                version_modele=version,
            )
        )
    return resultats


@app.post("/v1/qualifier", response_model=Qualification, tags=["qualification"])
def qualifier(demande: DemandeAQualifier, _cle: str = Depends(authentifier)):
    """Qualifie une demande : métier, urgence, confiance, explication, besoin de revue humaine."""
    return _qualifier([demande])[0]


@app.post("/v1/qualifier/lot", response_model=Lot, tags=["qualification"])
def qualifier_lot(lot: LotAQualifier, _cle: str = Depends(authentifier)):
    """Qualifie jusqu'à 100 demandes en un appel (utilisé par le service de qualification)."""
    return Lot(resultats=_qualifier(lot.demandes))
