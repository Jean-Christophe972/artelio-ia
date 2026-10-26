"""Surveillance du modèle en production : dérive des données (Evidently) et performance réelle.

    python monitoring/surveillance.py            # une analyse (lancée chaque heure par le planificateur)

1. Référence : prédictions du champion sur les données récentes du jeu d'entraînement.
   Courant : prédictions réellement servies par l'API sur les dernières 24 h (ml.predictions).
2. Evidently compare la distribution de 8 variables : longueur du texte, nombre de mots, part de
   mots inconnus du vocabulaire, confiance métier et urgence, métier et urgence prédits, canal.
   On mesure l'AMPLEUR de l'écart, pas sa significativité statistique : avec des milliers de
   prédictions, un test du khi-deux ou de Kolmogorov-Smirnov déclare « dérive » pour des écarts
   minuscules (constaté : 55,3 % contre 53,6 % de demandes via l'application → p ≈ 0).
     - variables catégorielles : PSI (Population Stability Index), dérive si PSI > 0,2 ;
     - variables numériques : distance de Wasserstein normalisée, dérive si > 0,1.
   Une dérive globale est déclarée quand plus de 30 % des variables ont dérivé.
3. Performance réelle : part des métiers prédits que le support a corrigés (7 derniers jours).
4. Le rapport HTML est archivé dans le bucket mlflow, le résultat dans ml.rapports_derive, et les
   métriques sont publiées pour Prometheus (alerte DeriveDonneesDetectee). Une dérive déclenche
   le réentraînement automatique (retrain/planificateur.py).
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import psycopg

from artelio_ia import donnees as d
from artelio_ia import registre
from artelio_ia.config import env

log = logging.getLogger("surveillance")
NUMERIQUES = ["longueur", "nb_mots", "taux_hors_vocabulaire", "proba_metier", "proba_urgence"]
CATEGORIELLES = ["metier", "urgence", "canal"]
SEUIL_PART_DERIVE = 0.3
MINIMUM_COURANT = 200
FENETRE_COURANTE = 1000  # les 1 000 dernières prédictions des dernières 24 h (fenêtre glissante)


def connexion() -> psycopg.Connection:
    return psycopg.connect(
        host=env("PG_HOST"),
        dbname=env("PG_DATABASE", "artelio"),
        user=env("PG_USER", "ml_service"),
        password=env("PG_PASSWORD"),
        sslmode="require",
        connect_timeout=5,
    )


def variables(qualifieur, textes: list[str], canaux: list[str]) -> pd.DataFrame:
    """Mêmes variables que celles journalisées par l'API (voir api/service.py)."""
    from api.service import taux_hors_vocabulaire

    preds = qualifieur.predire(textes)
    return pd.DataFrame(
        {
            "longueur": [len(t) for t in textes],
            "nb_mots": [len(t.split()) for t in textes],
            "taux_hors_vocabulaire": [taux_hors_vocabulaire(qualifieur, t) for t in textes],
            "proba_metier": [p.proba_metier for p in preds],
            "proba_urgence": [p.proba_urgence for p in preds],
            "metier": [p.metier for p in preds],
            "urgence": [p.urgence for p in preds],
            "canal": canaux,
        }
    )


def reference(qualifieur, taille: int = 1000) -> pd.DataFrame:
    jeu = d.charger(qualifieur.metadonnees.get("version_donnees"))
    _, recent = d.decouper_temporel(jeu.donnees)
    echantillon = recent.sample(min(taille, len(recent)), random_state=0)
    return variables(qualifieur, echantillon["texte"].tolist(), echantillon["canal"].tolist())


def analyser() -> dict:
    from evidently.metric_preset import DataDriftPreset
    from evidently.pipeline.column_mapping import ColumnMapping
    from evidently.report import Report

    qualifieur = registre.charger_depuis_registre("champion")
    with connexion() as conn:
        courant = pd.DataFrame(
            conn.execute(
                f"""SELECT {", ".join(NUMERIQUES + CATEGORIELLES)} FROM ml.predictions
                    WHERE horodatage > now() - interval '24 hours' ORDER BY horodatage DESC LIMIT %s""",
                (FENETRE_COURANTE,),
            ).fetchall(),
            columns=NUMERIQUES + CATEGORIELLES,
        )
        corrections = conn.execute(
            """SELECT count(*) FILTER (WHERE metier_valide IS NOT NULL AND metier_valide <> metier_predit)::real
                      / nullif(count(*), 0)
               FROM app.demandes WHERE metier_predit IS NOT NULL AND cree_le > now() - interval '7 days'"""
        ).fetchone()[0]
        conn.execute("DELETE FROM ml.predictions WHERE horodatage < now() - interval '90 days'")  # durée de conservation
        conn.commit()

    if len(courant) < MINIMUM_COURANT:
        log.info("Seulement %d prédictions sur 24 h (minimum %d) : analyse reportée", len(courant), MINIMUM_COURANT)
        return {"analyse": False, "n_courant": len(courant)}

    ref = reference(qualifieur)
    rapport = Report(
        metrics=[
            DataDriftPreset(
                drift_share=SEUIL_PART_DERIVE,
                cat_stattest="psi",
                cat_stattest_threshold=0.2,
                num_stattest="wasserstein",
                num_stattest_threshold=0.1,
            )
        ]
    )
    rapport.run(
        reference_data=ref,
        current_data=courant,
        column_mapping=ColumnMapping(numerical_features=NUMERIQUES, categorical_features=CATEGORIELLES),
    )
    resultat = rapport.as_dict()["metrics"]
    synthese = resultat[0]["result"]
    detail = resultat[1]["result"]["drift_by_columns"]
    derivees = sorted(c for c, v in detail.items() if v["drift_detected"])
    derive = bool(synthese["dataset_drift"])

    horodatage = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    cle = f"rapports-derive/{horodatage}.html"
    with tempfile.TemporaryDirectory() as tmp:
        chemin = Path(tmp) / "rapport.html"
        rapport.save_html(str(chemin))
        d._s3().upload_file(str(chemin), "mlflow", cle, ExtraArgs={"ContentType": "text/html"})

    with connexion() as conn:
        conn.execute(
            """INSERT INTO ml.rapports_derive (version_modele, n_reference, n_courant, derive_detectee, part_variables,
                   variables_derivees, taux_correction, cle_rapport) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                qualifieur.metadonnees.get("version_registre"),
                len(ref),
                len(courant),
                derive,
                synthese["share_of_drifted_columns"],
                derivees,
                corrections,
                cle,
            ),
        )
        conn.commit()

    publier(derive, synthese["share_of_drifted_columns"], corrections)
    log.info(
        "Dérive %s : %.0f %% des variables (%s) ; taux de correction humaine %s ; rapport s3://mlflow/%s",
        "DÉTECTÉE" if derive else "non détectée",
        100 * synthese["share_of_drifted_columns"],
        ", ".join(derivees) or "aucune",
        f"{corrections:.1%}" if corrections is not None else "n/a",
        cle,
    )
    return {"analyse": True, "derive": derive, "variables_derivees": derivees, "rapport": cle}


def publier(derive: bool, part: float, corrections: float | None) -> None:
    passerelle = os.environ.get("PUSHGATEWAY")
    if not passerelle:
        return
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    reg = CollectorRegistry()
    Gauge("artelio_ia_derive_detectee", "1 si Evidently détecte une dérive", registry=reg).set(int(derive))
    Gauge("artelio_ia_derive_part_variables", "Part des variables en dérive", registry=reg).set(part)
    if corrections is not None:
        Gauge("artelio_ia_taux_correction_humaine", "Part des métiers prédits corrigés par le support", registry=reg).set(corrections)
    Gauge("artelio_ia_derniere_surveillance_timestamp_seconds", "Dernière analyse", registry=reg).set_to_current_time()
    try:
        push_to_gateway(passerelle, job="artelio_ia_surveillance", registry=reg, timeout=5)
    except OSError as exc:
        log.warning("Pushgateway injoignable : %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s — %(message)s")
    print(analyser())
    sys.exit(0)
