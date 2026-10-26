"""Planificateur MLOps : surveillance horaire et réentraînement automatique.

Toutes les 10 minutes, il vérifie s'il faut :
  - lancer la surveillance de la dérive (toutes les heures) ;
  - réentraîner le modèle, pour l'une de ces raisons :
      1. « nouvelle_version » : le pipeline (bloc 3) a publié un jeu plus récent que celui du champion ;
      2. « derive »           : la dernière analyse Evidently a détecté une dérive depuis le dernier entraînement ;
      3. « hebdomadaire »     : le champion a plus de 7 jours (filet de sécurité).
Le réentraînement passe TOUJOURS par la barrière de promotion (entrainement.py) : un modèle
moins bon que le champion est enregistré comme « challenger » mais n'est pas mis en service.
L'API charge automatiquement le nouveau champion (rechargement à chaud).
"""

from __future__ import annotations

import datetime as dt
import logging
import subprocess
import sys
import time
from pathlib import Path

from artelio_ia import donnees as d
from artelio_ia import registre

log = logging.getLogger("planificateur")
RACINE = Path(__file__).resolve().parents[1]
INTERVALLE = 600
PERIODE_SURVEILLANCE = dt.timedelta(hours=1)
AGE_MAXIMAL = dt.timedelta(days=7)


def derive_depuis(instant: dt.datetime) -> bool:
    from monitoring.surveillance import connexion

    with connexion() as conn:
        row = conn.execute("SELECT bool_or(derive_detectee) FROM ml.rapports_derive WHERE horodatage > %s", (instant,)).fetchone()
    return bool(row and row[0])


def derniere_tentative() -> dt.datetime | None:
    """Date du dernier entraînement, PROMU OU NON (sinon un modèle refusé relancerait une boucle)."""
    import mlflow

    runs = mlflow.search_runs(experiment_names=["qualification-demandes"], order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        return None
    return runs.iloc[0]["start_time"].to_pydatetime()


def raison_de_reentrainer() -> str | None:
    champion = registre.charger_depuis_registre("champion")
    if champion is None:
        return "initial"
    meta = champion.metadonnees
    reference = max(filter(None, [dt.datetime.fromisoformat(meta["entraine_le"]), derniere_tentative()]))
    if d.derniere_version() != meta.get("version_donnees") and dt.datetime.now(dt.UTC) - reference > dt.timedelta(hours=12):
        return "nouvelle_version"
    if derive_depuis(reference):
        return "derive"
    if dt.datetime.now(dt.UTC) - reference > AGE_MAXIMAL:
        return "hebdomadaire"
    return None


def reentrainer(raison: str) -> None:
    log.info("Réentraînement déclenché : %s", raison)
    commande = [sys.executable, "-m", "artelio_ia.entrainement", "--declencheur", raison, "--mode", "reentrainement"]
    resultat = subprocess.run(commande, cwd=RACINE, check=False)
    log.info("Réentraînement terminé (code %s)", resultat.returncode)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s — %(message)s")
    derniere_surveillance = dt.datetime.min.replace(tzinfo=dt.UTC)
    while True:
        try:
            if dt.datetime.now(dt.UTC) - derniere_surveillance >= PERIODE_SURVEILLANCE:
                from monitoring.surveillance import analyser

                analyser()
                derniere_surveillance = dt.datetime.now(dt.UTC)
            if raison := raison_de_reentrainer():
                reentrainer(raison)
        except Exception:  # noqa: BLE001 — le planificateur ne doit jamais s'arrêter sur une erreur ponctuelle
            log.exception("Cycle du planificateur en échec")
        time.sleep(INTERVALLE)


if __name__ == "__main__":
    sys.exit(main())
