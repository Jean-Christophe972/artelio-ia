"""Service de qualification : relie l'IA au flux de la marketplace.

Toutes les 5 secondes, il prend les nouvelles demandes arrivées par le pipeline temps réel
(statut « nouvelle »), les fait qualifier par l'API (en lot), puis écrit le métier, l'urgence
et la confiance dans la base. Toutes les prédictions passent donc par l'API : elles sont
mesurées, journalisées et surveillées (dérive) au même endroit.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.request

import psycopg

from artelio_ia.config import env

log = logging.getLogger("qualification")


def appeler_api(demandes: list[tuple]) -> list[dict]:
    corps = json.dumps(
        {"demandes": [{"texte": texte, "canal": canal, "demande_id": str(id_), "expliquer": False} for id_, texte, canal in demandes]}
    ).encode()
    requete = urllib.request.Request(
        env("API_URL", "http://artelio-api:8000") + "/v1/qualifier/lot",
        data=corps,
        headers={"Content-Type": "application/json", "X-API-Key": env("API_KEY")},
    )
    with urllib.request.urlopen(requete, timeout=30) as reponse:
        return json.loads(reponse.read())["resultats"]


def cycle(conn: psycopg.Connection, taille: int = 100) -> int:
    demandes = conn.execute(
        "SELECT id, description, canal FROM app.demandes WHERE statut = 'nouvelle' AND metier_predit IS NULL ORDER BY cree_le LIMIT %s",
        (taille,),
    ).fetchall()
    if not demandes:
        return 0
    resultats = appeler_api(demandes)
    with conn.cursor() as cur:
        cur.executemany(
            """UPDATE app.demandes SET metier_predit = %(metier)s, urgence_predite = %(urgence)s,
                   confiance_ia = %(confiance_metier)s, statut = 'qualifiee'
               WHERE id = %(demande_id)s AND statut = 'nouvelle'""",
            resultats,
        )
    conn.commit()
    revue = sum(r["revue_humaine"] for r in resultats)
    log.info("%d demandes qualifiées (%d en revue humaine)", len(resultats), revue)
    return len(resultats)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s — %(message)s")
    intervalle = float(os.environ.get("INTERVALLE_S", "5"))
    conn = None
    while True:
        try:
            conn = (
                conn
                if conn and not conn.closed
                else psycopg.connect(
                    host=env("PG_HOST"),
                    dbname=env("PG_DATABASE", "artelio"),
                    user=env("PG_USER", "ml_service"),
                    password=env("PG_PASSWORD"),
                    sslmode="require",
                    connect_timeout=5,
                )
            )
            while cycle(conn) == 100:  # file en retard : on enchaîne les lots sans attendre
                pass
        except Exception:  # noqa: BLE001 — l'API ou la base peut être momentanément indisponible
            log.exception("Cycle de qualification en échec, nouvel essai dans %ss", intervalle)
            if conn:
                conn.close()
            conn = None
        time.sleep(intervalle)


if __name__ == "__main__":
    sys.exit(main())
