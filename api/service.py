"""Gestion du modèle servi : chargement, rechargement à chaud, journal des prédictions."""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from pathlib import Path

from artelio_ia import registre
from artelio_ia.config import DOSSIER_MODELES
from artelio_ia.modele import Qualifieur
from artelio_ia.pii import masquer

log = logging.getLogger("api.service")


class ModeleServi:
    """Détient le modèle courant. Source : registre MLflow (alias « champion ») ou dossier local.

    En mode registre, un fil d'arrière-plan vérifie l'alias toutes les 60 s : quand le
    réentraînement promeut un nouveau champion, l'API le charge SANS redémarrer ni couper le
    service (l'ancien modèle répond jusqu'à ce que le nouveau soit prêt).
    """

    def __init__(self):
        self.source = os.environ.get("MODELE_SOURCE", "local")
        self.qualifieur: Qualifieur | None = None
        self.version = "aucune"
        self._verrou = threading.Lock()

    def charger(self) -> None:
        if self.source == "registre":
            version = registre.version_alias("champion")
            if version is None:
                raise RuntimeError("aucun modèle « champion » dans le registre MLflow")
            qualifieur = registre.charger_depuis_registre("champion")
            nom_version = f"registre-v{version}"
        else:
            dossier = Path(os.environ.get("MODELE_DOSSIER", DOSSIER_MODELES / "champion"))
            qualifieur = registre.charger_local(dossier)
            nom_version = f"local-{qualifieur.metadonnees.get('entraine_le', '?')[:19]}"
        qualifieur.predire(["préchauffage du modèle"])  # charge le réseau de plongements avant le 1er client
        with self._verrou:
            self.qualifieur, self.version = qualifieur, nom_version
        log.info("Modèle chargé : %s", nom_version)

    def surveiller_registre(self, intervalle: int = 60) -> None:
        if self.source != "registre":
            return

        def boucle():
            while True:
                time.sleep(intervalle)
                try:
                    version = registre.version_alias("champion")
                    if version and f"registre-v{version}" != self.version:
                        log.info("Nouveau champion détecté (v%s) : rechargement à chaud", version)
                        self.charger()
                except Exception:  # noqa: BLE001 — le service continue avec le modèle courant
                    log.exception("Échec du rechargement ; le modèle courant reste en service")

        threading.Thread(target=boucle, daemon=True, name="rechargement-modele").start()


def taux_hors_vocabulaire(qualifieur: Qualifieur, texte: str) -> float:
    """Part des mots inconnus du vocabulaire d'entraînement : indicateur précoce de dérive (voir Qualifieur)."""
    return qualifieur.taux_hors_vocabulaire(texte)


class JournalPredictions:
    """Enregistre chaque prédiction dans ml.predictions, en arrière-plan (sans ralentir la réponse).

    Le texte est MASQUÉ (téléphones, emails, IBAN) avant stockage ; conservation 90 jours.
    Sert à la surveillance de la dérive et à l'audit des décisions de l'IA (AI Act).
    """

    SCHEMA = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")

    def __init__(self):
        self.actif = bool(os.environ.get("PG_HOST"))
        self.file: queue.Queue = queue.Queue(maxsize=10000)
        if self.actif:
            threading.Thread(target=self._ecrire, daemon=True, name="journal-predictions").start()

    def ajouter(self, ligne: dict) -> None:
        if not self.actif:
            return
        try:
            self.file.put_nowait(ligne)
        except queue.Full:
            log.warning("File du journal pleine : prédiction non journalisée")

    def _connexion(self):
        import psycopg

        conn = psycopg.connect(
            host=os.environ["PG_HOST"],
            dbname=os.environ.get("PG_DATABASE", "artelio"),
            user=os.environ.get("PG_USER", "ml_service"),
            password=os.environ["PG_PASSWORD"],
            sslmode="require",
            autocommit=True,
            connect_timeout=5,
        )
        conn.execute(self.SCHEMA)
        return conn

    def _ecrire(self) -> None:
        conn = None
        while True:
            lot = [self.file.get()]
            while not self.file.empty() and len(lot) < 200:
                lot.append(self.file.get_nowait())
            try:
                conn = conn if conn and not conn.closed else self._connexion()
                with conn.cursor() as cur:
                    cur.executemany(
                        """INSERT INTO ml.predictions (demande_id, version_modele, texte_masque, longueur, nb_mots,
                               taux_hors_vocabulaire, canal, metier, proba_metier, urgence, proba_urgence, revue_humaine,
                               motifs, latence_ms)
                           VALUES (%(demande_id)s, %(version_modele)s, %(texte_masque)s, %(longueur)s, %(nb_mots)s,
                               %(taux_hors_vocabulaire)s, %(canal)s, %(metier)s, %(proba_metier)s, %(urgence)s,
                               %(proba_urgence)s, %(revue_humaine)s, %(motifs)s, %(latence_ms)s)""",
                        lot,
                    )
            except Exception:  # noqa: BLE001 — la journalisation ne doit jamais faire tomber l'API
                log.exception("Échec de la journalisation de %d prédictions", len(lot))
                conn = None
                time.sleep(2)


def ligne_journal(texte: str, canal: str, demande_id, prediction, version: str, hors_vocab: float, latence_ms: float) -> dict:
    texte_masque, _ = masquer(texte)
    return {
        "demande_id": demande_id,
        "version_modele": version,
        "texte_masque": texte_masque,
        "longueur": len(texte),
        "nb_mots": len(re.findall(r"\w+", texte)),
        "taux_hors_vocabulaire": round(hors_vocab, 4),
        "canal": canal,
        "metier": prediction.metier,
        "proba_metier": prediction.proba_metier,
        "urgence": prediction.urgence,
        "proba_urgence": prediction.proba_urgence,
        "revue_humaine": prediction.revue_humaine,
        "motifs": prediction.motifs_revue,
        "latence_ms": round(latence_ms, 2),
    }
