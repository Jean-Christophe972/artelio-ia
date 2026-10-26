"""Entraînement, évaluation, enregistrement et promotion du modèle.

    python -m artelio_ia.entrainement                       # dernière version du jeu, registre MLflow
    python -m artelio_ia.entrainement --fichier data.csv    # fichier local (tests, CI)
    python -m artelio_ia.entrainement --exporter            # exporte aussi le champion dans models/champion

Étapes :
  1. données : version publiée par le pipeline (empreinte vérifiée), découpage TEMPOREL
  2. sélection de modèle : validation croisée stratifiée à 5 plis sur la partie entraînement
  3. entraînement final, puis évaluation sur 2 jeux jamais vus :
       - test temporel (les 60 derniers jours) ;
       - test hors gabarits (écrit à la main, formulations différentes)
  4. robustesse, performance par canal, calibration, latence
  5. MLflow : paramètres, métriques, artefacts, version du modèle
  6. barrière de promotion : le challenger ne devient champion que s'il fait au moins aussi
     bien que le champion actuel sur les MÊMES jeux de test, et respecte les planchers métier
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from artelio_ia import augmentation, plongements, registre
from artelio_ia import donnees as d
from artelio_ia import evaluation as ev
from artelio_ia.config import (
    DOSSIER_MODELES,
    GRAINE,
    MACRO_F1_METIER_MIN,
    N_AUGMENTATION,
    NOM_MODELE,
    RAPPEL_CRITIQUE_MIN,
    SEUIL_ALERTE_CRITIQUE,
    SEUIL_CONFIANCE_METIER,
    TOLERANCE_REGRESSION,
)
from artelio_ia.modele import Qualifieur, candidats

log = logging.getLogger("entrainement")


def selectionner(train: pd.DataFrame, validation: pd.DataFrame, cible: str, plis: int = 5) -> tuple[str, pd.DataFrame, dict]:
    """Compare les candidats et choisit le meilleur sur le jeu de VALIDATION hors gabarits.

    La validation croisée est rapportée mais ne départage pas : sur des données issues de gabarits,
    tous les modèles y obtiennent ~1,00 (ils reconnaissent les gabarits). Le choix se fait donc sur
    des textes formulés autrement ; le jeu de test final, lui, n'intervient jamais.
    """
    cv = StratifiedKFold(n_splits=plis, shuffle=True, random_state=GRAINE)
    lignes, ajustes = [], {}
    for nom, pipeline in candidats().items():
        lourd = "plongements" in nom or "hybride" in nom
        scores = cross_val_score(pipeline, train["texte"], train[cible], cv=cv, scoring="f1_macro", n_jobs=1 if lourd else 2)
        modele = clone(pipeline).fit(train["texte"], train[cible])
        f1_validation = f1_score(validation[cible], modele.predict(validation["texte"]), average="macro")
        ajustes[nom] = modele
        lignes.append({"modele": nom, "f1_macro_cv": scores.mean(), "ecart_type_cv": scores.std(), "f1_macro_validation": f1_validation})
        log.info("[%s] %-38s CV %.4f ± %.4f | validation hors gabarits %.4f", cible, nom, scores.mean(), scores.std(), f1_validation)
    tableau = pd.DataFrame(lignes).sort_values("f1_macro_validation", ascending=False)
    # Candidats retenus : ceux qui fournissent des probabilités (revue humaine, explicabilité)
    probabilistes = tableau[tableau["modele"].str.contains("regression_logistique")]
    meilleur = probabilistes.iloc[0]["modele"]
    return meilleur, tableau, ajustes[meilleur]


def entrainer(train: pd.DataFrame, validation: pd.DataFrame, plis: int = 5) -> tuple[Qualifieur, dict]:
    """Mode « selection » (expérimentation, manuel) : compare tous les candidats et choisit."""
    choix, tableaux, modeles = {}, {}, {}
    for cible in ("metier", "urgence"):
        choix[cible], tableaux[cible], modeles[cible] = selectionner(train, validation, cible, plis)
    return Qualifieur(modeles["metier"], modeles["urgence"]), {"choix": choix, "tableaux": tableaux}


def reentrainer_configuration(train: pd.DataFrame, choix: dict[str, str]) -> tuple[Qualifieur, dict]:
    """Mode « reentrainement » (automatique) : même configuration que le champion, données nouvelles.

    La sélection de modèle est une étape d'expérimentation ; en production on réentraîne la
    configuration validée (rapide, prévisible, reproductible), puis la barrière de promotion décide.
    """
    modeles = {cible: clone(candidats()[nom]).fit(train["texte"], train[cible]) for cible, nom in choix.items()}
    return Qualifieur(modeles["metier"], modeles["urgence"]), {"choix": choix, "tableaux": {}}


def evaluer_complet(
    qualifieur: Qualifieur, test_temporel: pd.DataFrame, hors_gabarits: pd.DataFrame, test_final: pd.DataFrame | None = None
) -> dict:
    """hors_gabarits = jeu de VALIDATION (décisions) ; test_final = mesure seulement, jamais utilisé pour décider."""
    return {
        "test_temporel": ev.metriques(qualifieur, test_temporel) if len(test_temporel) else None,
        "hors_gabarits": ev.metriques(qualifieur, hors_gabarits),
        "test_final": ev.metriques(qualifieur, test_final) if test_final is not None else None,
        "robustesse": ev.robustesse(qualifieur, hors_gabarits).to_dict(orient="records"),
        "par_canal": ev.par_tranche(qualifieur, hors_gabarits, "canal").to_dict(orient="records"),
        "couverture": ev.compromis_couverture(qualifieur, hors_gabarits).to_dict(orient="records"),
        "latence_ms": qualifieur.latence_ms(),
        "ablation_sans_garde_fous": ablation(qualifieur, hors_gabarits),
    }


def ablation(qualifieur: Qualifieur, donnees: pd.DataFrame) -> dict:
    """Apport des garde-fous : mêmes métriques d'urgence, règles désactivées."""
    qualifieur.garde_fous = False
    try:
        sans = ev.metriques(qualifieur, donnees)["urgence"]
    finally:
        qualifieur.garde_fous = True
    return {k: sans[k] for k in ("exactitude", "f1_macro", "rappel_critique", "sous_estimation_grave")}


def decider_promotion(challenger: dict, champion: dict | None) -> tuple[bool, list[str]]:
    """Barrière de promotion. Renvoie (promu, raisons)."""
    raisons = []
    hg = challenger["hors_gabarits"]
    if hg["metier"]["f1_macro"] < MACRO_F1_METIER_MIN:
        raisons.append(f"F1 macro métier hors gabarits {hg['metier']['f1_macro']:.3f} < plancher {MACRO_F1_METIER_MIN}")
    if hg["urgence"]["rappel_critique"] < RAPPEL_CRITIQUE_MIN:
        raisons.append(f"rappel des urgences critiques {hg['urgence']['rappel_critique']:.3f} < plancher {RAPPEL_CRITIQUE_MIN}")
    if champion is not None:
        for jeu in ("hors_gabarits", "test_temporel"):
            if challenger[jeu] is None or champion[jeu] is None:
                continue
            for tache in ("metier", "urgence"):
                nouveau, ancien = challenger[jeu][tache]["f1_macro"], champion[jeu][tache]["f1_macro"]
                if nouveau < ancien - TOLERANCE_REGRESSION:
                    raisons.append(f"régression {tache} sur {jeu} : {nouveau:.3f} contre {ancien:.3f} pour le champion")
    return not raisons, raisons


def resume(resultats: dict) -> dict[str, float]:
    """Métriques à plat pour MLflow et Prometheus."""
    plat = {"latence_ms": resultats["latence_ms"]}
    for jeu in ("test_temporel", "hors_gabarits", "test_final"):
        r = resultats.get(jeu)
        if r is None:
            continue
        for tache in ("metier", "urgence"):
            plat[f"{jeu}.{tache}.f1_macro"] = r[tache]["f1_macro"]
            plat[f"{jeu}.{tache}.exactitude"] = r[tache]["exactitude"]
        plat[f"{jeu}.metier.ece"] = r["metier"]["ece"]
        plat[f"{jeu}.urgence.rappel_critique"] = r["urgence"]["rappel_critique"]
        plat[f"{jeu}.urgence.sous_estimation_grave"] = r["urgence"]["sous_estimation_grave"]
        plat[f"{jeu}.taux_revue_humaine"] = r["revue_humaine"]["taux"]
    for k, v in resultats["ablation_sans_garde_fous"].items():
        plat[f"hors_gabarits.urgence_sans_garde_fous.{k}"] = v
    return plat


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s — %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", help="version du jeu de données (défaut : la dernière publiée)")
    parser.add_argument("--fichier", type=Path, help="fichier CSV local au lieu de S3")
    parser.add_argument("--plis", type=int, default=5)
    parser.add_argument("--exporter", action="store_true", help="si promu, exporte le modèle dans models/champion")
    parser.add_argument("--declencheur", default="manuel", help="cause de l'entraînement (manuel, nouvelle_version, derive…)")
    parser.add_argument(
        "--mode",
        choices=["selection", "reentrainement"],
        default="selection",
        help="selection : compare tous les candidats ; reentrainement : configuration du champion",
    )
    args = parser.parse_args()
    np.random.seed(GRAINE)

    # 1. Données
    if args.fichier:
        donnees = pd.read_csv(args.fichier, parse_dates=["cree_le"])
        d.valider_schema(donnees)
        version, sha = args.fichier.stem, registre.sha256_fichier(args.fichier)
    else:
        jeu = d.charger(args.version)
        donnees, version, sha = jeu.donnees, jeu.version, jeu.sha256
    train, test_temporel = d.decouper_temporel(donnees)
    hors_gabarits = d.charger_hors_gabarits("validation")
    test_final = d.charger_hors_gabarits("test")
    enrichi = augmentation.generer(N_AUGMENTATION) if N_AUGMENTATION else pd.DataFrame(columns=["texte", "metier", "urgence"])
    train = pd.concat([train.assign(source="pipeline"), enrichi], ignore_index=True)
    train = pd.concat([train, augmentation.copies_bruitees(train)], ignore_index=True)  # robustesse aux fautes de frappe
    log.info(
        "Train : %d (dont %d enrichis par le lexique) | test temporel : %d | validation hors gabarits : %d",
        len(train),
        len(enrichi),
        len(test_temporel),
        len(hors_gabarits),
    )

    # 2-3. Sélection et entraînement
    plongements.activer_cache(True)  # accélère la validation croisée
    champion_actuel = registre.charger_depuis_registre("champion") if args.mode == "reentrainement" else None
    if champion_actuel is not None:
        qualifieur, selection = reentrainer_configuration(train, champion_actuel.metadonnees["modeles"])
    else:
        qualifieur, selection = entrainer(train, hors_gabarits, args.plis)
    plongements.activer_cache(False)  # mesures suivantes (dont la latence) dans les conditions réelles
    # 4. Évaluation
    resultats = evaluer_complet(qualifieur, test_temporel, hors_gabarits, test_final)
    metriques = resume(resultats)
    for cle, valeur in metriques.items():
        log.info("%-42s %.4f", cle, valeur)

    qualifieur.metadonnees = {
        "nom": NOM_MODELE,
        "entraine_le": dt.datetime.now(dt.UTC).isoformat(),
        "declencheur": args.declencheur,
        "mode": args.mode,
        "version_donnees": version,
        "sha256_donnees": sha,
        "n_train": len(train),
        "n_enrichissement": int((train["source"] == "lexique").sum()),
        "n_copies_bruitees": int((train["source"] == "bruit").sum()),
        "modeles": selection["choix"],
        "seuils": {"confiance_metier": SEUIL_CONFIANCE_METIER, "alerte_critique": SEUIL_ALERTE_CRITIQUE},
        "metriques": metriques,
        "scikit_learn": sklearn.__version__,
        "commit_git": os.environ.get("GIT_SHA", "inconnu"),
        "graine": GRAINE,
    }

    # 5-6. Registre et promotion
    promu, raisons = True, []
    with tempfile.TemporaryDirectory() as tmp:
        dossier = Path(tmp) / "modele"
        chemin = registre.sauvegarder(qualifieur, dossier)
        if os.environ.get("MLFLOW_TRACKING_URI"):
            promu, raisons = enregistrer_mlflow(qualifieur, chemin, dossier, selection, resultats, metriques, test_temporel, hors_gabarits)
        else:
            promu, raisons = decider_promotion(resultats, None)
            log.info("MLFLOW_TRACKING_URI absent : pas de registre (mode local), planchers seuls appliqués")
        if promu and args.exporter:
            cible = DOSSIER_MODELES / "champion"
            registre.sauvegarder(qualifieur, cible)
            (cible / "evaluation.json").write_text(json.dumps(resultats, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
            log.info("Champion exporté dans %s", cible)

    publier_metriques(metriques, promu)
    print(json.dumps({"version_donnees": version, "promu": promu, "raisons_refus": raisons, **metriques}, indent=2, ensure_ascii=False))
    return 0


def enregistrer_mlflow(qualifieur, chemin, dossier, selection, resultats, metriques, test_temporel, hors_gabarits):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow
    from mlflow import MlflowClient
    from sklearn.metrics import ConfusionMatrixDisplay

    mlflow.set_experiment("qualification-demandes")
    client = MlflowClient()
    with mlflow.start_run(run_name=f"entrainement-{qualifieur.metadonnees['version_donnees']}") as run:
        meta = qualifieur.metadonnees
        mlflow.set_tags({"declencheur": meta["declencheur"], "version_donnees": meta["version_donnees"], "commit_git": meta["commit_git"]})
        mlflow.log_params(
            {
                "version_donnees": meta["version_donnees"],
                "sha256_donnees": meta["sha256_donnees"],
                "n_train": meta["n_train"],
                "modele_metier": meta["modeles"]["metier"],
                "modele_urgence": meta["modeles"]["urgence"],
                "seuil_confiance_metier": SEUIL_CONFIANCE_METIER,
                "seuil_alerte_critique": SEUIL_ALERTE_CRITIQUE,
                "graine": GRAINE,
            }
        )
        mlflow.log_metrics({k.replace(".", "_"): v for k, v in metriques.items()})
        for cible, tableau in selection["tableaux"].items():
            mlflow.log_table(tableau, f"selection/validation_croisee_{cible}.json")
        mlflow.log_dict(json.loads(json.dumps(resultats, default=float)), "evaluation/resultats.json")
        for jeu, donnees_jeu in (("test_temporel", test_temporel), ("hors_gabarits", hors_gabarits)):
            if len(donnees_jeu) == 0:
                continue
            preds = qualifieur.predire(donnees_jeu["texte"].tolist())
            for tache in ("metier", "urgence"):
                fig, ax = plt.subplots(figsize=(7, 6))
                ConfusionMatrixDisplay.from_predictions(
                    donnees_jeu[tache], [getattr(p, tache) for p in preds], ax=ax, xticks_rotation=45, colorbar=False
                )
                ax.set_title(f"{tache} — {jeu}")
                fig.tight_layout()
                mlflow.log_figure(fig, f"evaluation/confusion_{tache}_{jeu}.png")
                plt.close(fig)

        info = mlflow.pyfunc.log_model(
            artifact_path="modele",
            python_model=registre.ModeleArtelio(),
            artifacts={"qualifieur": str(chemin)},
            code_paths=[str(Path(__file__).parent)],
            input_example=pd.DataFrame({"texte": ["fuite d'eau sous l'évier"]}),
        )
        version = mlflow.register_model(info.model_uri, NOM_MODELE)
        sha = json.loads((dossier / "metadata.json").read_text())["sha256_artefact"]
        client.set_model_version_tag(NOM_MODELE, version.version, "sha256_artefact", sha)
        client.set_model_version_tag(NOM_MODELE, version.version, "version_donnees", meta["version_donnees"])

        # Barrière de promotion : on réévalue le champion actuel sur les MÊMES jeux de test
        champion = registre.charger_depuis_registre("champion")
        resultats_champion = None
        if champion is not None:
            resultats_champion = {
                "test_temporel": ev.metriques(champion, test_temporel) if len(test_temporel) else None,
                "hors_gabarits": ev.metriques(champion, hors_gabarits),
            }
        promu, raisons = decider_promotion(resultats, resultats_champion)
        alias = "champion" if promu else "challenger"
        client.set_registered_model_alias(NOM_MODELE, alias, version.version)
        client.set_model_version_tag(NOM_MODELE, version.version, "decision", "promu" if promu else "refuse : " + " ; ".join(raisons))
        mlflow.set_tag("decision_promotion", "promu" if promu else "refuse")
        log.info("Version %s → alias « %s »%s", version.version, alias, "" if promu else f" (refus : {raisons})")
        log.info("Run MLflow : %s", run.info.run_id)
    return promu, raisons


def publier_metriques(metriques: dict, promu: bool) -> None:
    passerelle = os.environ.get("PUSHGATEWAY")
    if not passerelle:
        return
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    reg = CollectorRegistry()
    Gauge("artelio_ia_dernier_entrainement_timestamp_seconds", "Fin du dernier entraînement", registry=reg).set_to_current_time()
    Gauge("artelio_ia_dernier_entrainement_promu", "1 si le dernier modèle entraîné a été promu", registry=reg).set(int(promu))
    g = Gauge("artelio_ia_evaluation", "Métriques d'évaluation du dernier modèle entraîné", ["metrique"], registry=reg)
    for cle, valeur in metriques.items():
        g.labels(cle).set(valeur)
    try:
        push_to_gateway(passerelle, job="artelio_ia_entrainement", registry=reg, timeout=5)
    except OSError as exc:
        log.warning("Pushgateway injoignable : %s", exc)


if __name__ == "__main__":
    sys.exit(main())
