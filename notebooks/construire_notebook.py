"""Construit le notebook d'analyse (01_exploration_modelisation.ipynb) à partir de cellules versionnées.

    python notebooks/construire_notebook.py && jupyter nbconvert --execute --to notebook --inplace notebooks/01_exploration_modelisation.ipynb

Écrire le notebook en Python garde son contenu lisible dans Git (revue de code) ; il est exécuté
dans l'image "ml" pour que les résultats affichés correspondent exactement au code.
"""

from pathlib import Path

import nbformat as nbf

CELLULES = [
    (
        "md",
        """# Qualification des demandes Artelio — exploration, modélisation, évaluation

**Objectif.** À partir du message d'un client (« fuite sous l'évier, ça coule partout »), prédire :
- le **métier** (7 classes) pour orienter vers le bon artisan ;
- l'**urgence** (4 niveaux ordonnés) pour prioriser l'intervention.

**Contraintes métier** (cahier des charges, `docs/CAHIER_DES_CHARGES_IA.md`) : ne jamais sous-estimer gravement une
urgence, envoyer en revue humaine ce qui est incertain, expliquer la décision (AI Act), répondre en moins de 300 ms
sur CPU.

**Plan.** 1. Données · 2. Le piège des gabarits · 3. Démarche v1 → v3 · 4. Évaluation du champion (intervalles de
confiance) · 5. Calibration et revue humaine · 6. Robustesse et équité · 7. Explicabilité · 8. Analyse d'erreurs · 9. Limites""",
    ),
    (
        "code",
        """import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score
from artelio_ia import donnees as d, evaluation as ev, registre
from artelio_ia.config import DOSSIER_MODELES, URGENCES, GRAINE
from artelio_ia.modele import candidats
pd.set_option("display.precision", 3); plt.rcParams["figure.dpi"] = 110

jeu = d.charger()                                   # dernière version publiée par le pipeline (bloc 3), empreinte vérifiée
donnees = jeu.donnees
train, test_temporel = d.decouper_temporel(donnees)
validation, test_final = d.charger_hors_gabarits("validation"), d.charger_hors_gabarits("test")
print(f"Version {jeu.version} : {len(donnees)} demandes | train {len(train)} | test temporel {len(test_temporel)}")
print(f"Validation hors gabarits : {len(validation)} | test final hors gabarits : {len(test_final)}")""",
    ),
    ("md", "## 1. Les données"),
    (
        "code",
        """fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
donnees["metier"].value_counts().plot.barh(ax=axes[0], title="Métiers (déséquilibre ×4)")
donnees["urgence"].value_counts().reindex(URGENCES).plot.bar(ax=axes[1], title="Urgences", rot=0)
(donnees.assign(mois=donnees["cree_le"].dt.to_period("M")).groupby("mois")["metier"]
 .apply(lambda s: (s == "chauffage").mean()).plot(ax=axes[2], title="Part du chauffage par mois (saisonnalité)"))
plt.tight_layout(); plt.show()
donnees["texte"].str.split().str.len().describe().to_frame("nombre de mots").T""",
    ),
    (
        "md",
        """**À retenir.** Classes déséquilibrées (→ `class_weight="balanced"` et F1 **macro**, qui donne le même poids à chaque
classe), forte saisonnalité du chauffage (→ dérive attendue chaque hiver, d'où la surveillance et le réentraînement),
textes courts (~15 mots).""",
    ),
    (
        "md",
        """## 2. Le piège des gabarits

Les données du pipeline sont synthétiques, générées à partir de gabarits de phrases. Un modèle TF-IDF entraîné
dessus obtient un score parfait en validation croisée… et s'effondre sur des textes formulés autrement.""",
    ),
    (
        "code",
        """tfidf = candidats(avec_plongements=False)["regression_logistique_C4.0"]
cv = StratifiedKFold(5, shuffle=True, random_state=GRAINE)
lignes = []
for cible in ("metier", "urgence"):
    score_cv = cross_val_score(tfidf, train["texte"], train[cible], cv=cv, scoring="f1_macro", n_jobs=2).mean()
    modele = clone(tfidf).fit(train["texte"], train[cible])
    lignes.append({"tâche": cible, "F1 macro — validation croisée": score_cv,
                   "F1 macro — validation hors gabarits": f1_score(validation[cible], modele.predict(validation["texte"]), average="macro")})
pd.DataFrame(lignes)""",
    ),
    (
        "md",
        """La validation croisée mesure ici la capacité à **reconnaître les gabarits**, pas à comprendre une demande. D'où la
méthode d'évaluation retenue (détails dans `data/README.md`) :
- **validation hors gabarits** (143 textes écrits à la main) : sélection de modèle et barrière de promotion ;
- **test final hors gabarits** (90 textes, rédigés après le gel de la v2) : mesure uniquement, jamais pour décider.""",
    ),
    (
        "md",
        """## 3. Démarche : de la v1 à la v3

| Version | Changement | Pourquoi |
|---|---|---|
| v1 | TF-IDF (mots + caractères) + régression logistique, données du pipeline seules | référence |
| v2 | + enrichissement par **lexique métier**, + **garde-fous** déterministes sur les signaux de danger | diversité des formulations ; coût asymétrique des urgences ratées |
| v3 | + **plongements de phrases** (MiniLM multilingue) combinés au TF-IDF ; casse normalisée ; détection **hors domaine** ; copies **bruitées** | comprendre le sens sans mot commun ; robustesse |

Résultats sur le **test final** (mesures successives, toutes conservées dans MLflow) :""",
    ),
    (
        "code",
        """resultats_versions = pd.DataFrame([
    {"version": "v2 (TF-IDF + lexique + garde-fous)", "F1 métier": 0.800, "F1 urgence": 0.594, "rappel critique": 0.571, "sous-estimation grave": 0.044},
    {"version": "v3 (hybride TF-IDF + plongements)", "F1 métier": 0.785, "F1 urgence": 0.656, "rappel critique": 0.857, "sous-estimation grave": 0.000},
])
champion = registre.charger_local(DOSSIER_MODELES / "champion")
r = ev.metriques(champion, test_final)
resultats_versions.loc[len(resultats_versions)] = {"version": "champion livré (models/champion)", "F1 métier": r["metier"]["f1_macro"],
    "F1 urgence": r["urgence"]["f1_macro"], "rappel critique": r["urgence"]["rappel_critique"],
    "sous-estimation grave": r["urgence"]["sous_estimation_grave"]}
resultats_versions""",
    ),
    (
        "md",
        """## 4. Évaluation du champion, avec intervalles de confiance

Les jeux hors gabarits sont petits : un intervalle de confiance (bootstrap, 1 000 tirages) évite de sur-interpréter
quelques points d'écart.""",
    ),
    (
        "code",
        """def bootstrap(y, p, metrique, n=1000, graine=0):
    rng = np.random.default_rng(graine); y, p = np.asarray(y), np.asarray(p); scores = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); scores.append(metrique(y[i], p[i]))
    return np.percentile(scores, [2.5, 97.5])

def rappel_critique(y, p):
    m = y == "critique"; return (p[m] == "critique").mean() if m.any() else np.nan

lignes = []
for nom, jeu_eval in (("validation", validation), ("test final", test_final)):
    preds = champion.predire(jeu_eval["texte"].tolist())
    pm, pu = np.array([x.metier for x in preds]), np.array([x.urgence for x in preds])
    for libelle, y, p, f in (("F1 macro métier", jeu_eval["metier"], pm, lambda a, b: f1_score(a, b, average="macro")),
                             ("F1 macro urgence", jeu_eval["urgence"], pu, lambda a, b: f1_score(a, b, average="macro")),
                             ("rappel urgences critiques", jeu_eval["urgence"], pu, rappel_critique)):
        bas, haut = bootstrap(y, p, f)
        lignes.append({"jeu": nom, "métrique": libelle, "valeur": f(np.asarray(y), p), "IC 95 % bas": bas, "IC 95 % haut": haut})
pd.DataFrame(lignes)""",
    ),
    (
        "code",
        """preds = champion.predire(test_final["texte"].tolist())
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
ConfusionMatrixDisplay.from_predictions(test_final["metier"], [p.metier for p in preds], ax=axes[0], xticks_rotation=45, colorbar=False)
ConfusionMatrixDisplay.from_predictions(test_final["urgence"], [p.urgence for p in preds], labels=URGENCES, ax=axes[1], colorbar=False)
axes[0].set_title("Métier — test final"); axes[1].set_title("Urgence — test final"); plt.tight_layout(); plt.show()
pd.DataFrame(r["urgence"]["par_classe"]).T.loc[URGENCES, ["precision", "recall", "f1-score", "support"]]""",
    ),
    (
        "md",
        """**Lecture.** Les confusions d'urgence se font surtout entre niveaux **voisins** (normale ↔ haute) ; la sous-estimation
**grave** (critique classée normale ou faible) est l'erreur que la politique de décision cherche à éliminer.""",
    ),
    ("md", "## 5. Calibration et politique de revue humaine"),
    (
        "code",
        """probas = champion.modele_metier.predict_proba(validation["texte"].tolist())
confiance = probas.max(axis=1); correct = np.array(champion.modele_metier.classes_)[probas.argmax(1)] == validation["metier"].to_numpy()
tranches = np.linspace(0, 1, 11); centres, exactitudes = [], []
for bas, haut in zip(tranches[:-1], tranches[1:], strict=True):
    m = (confiance > bas) & (confiance <= haut)
    if m.sum() >= 3: centres.append(confiance[m].mean()); exactitudes.append(correct[m].mean())
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot([0, 1], [0, 1], "--", color="grey", label="calibration parfaite"); axes[0].plot(centres, exactitudes, "o-", label="champion")
axes[0].set(xlabel="confiance annoncée", ylabel="exactitude réelle", title=f"Diagramme de fiabilité (ECE = {ev.erreur_calibration(probas, validation['metier'].to_numpy(), list(champion.modele_metier.classes_)):.3f})"); axes[0].legend()
compromis = ev.compromis_couverture(champion, validation)
axes[1].plot(compromis["couverture_automatique"], compromis["exactitude_automatique"], "o-")
for _, ligne in compromis.iloc[::2].iterrows(): axes[1].annotate(f"{ligne.seuil:.2f}", (ligne.couverture_automatique, ligne.exactitude_automatique))
axes[1].set(xlabel="part traitée sans humain", ylabel="exactitude sur cette part", title="Compromis automatisation / exactitude (seuils)")
plt.tight_layout(); plt.show()""",
    ),
    (
        "md",
        """Le seuil de confiance (0,55) a été **fixé a priori par le métier puis mesuré** (et non optimisé sur le test) :
en dessous, la demande part en revue humaine. Les demandes traitées automatiquement sont nettement plus exactes que
celles envoyées en revue — c'est le but : l'humain se concentre là où le modèle doute.""",
    ),
    (
        "code",
        """r_val = ev.metriques(champion, validation)["revue_humaine"]
pd.Series({"part en revue humaine": r_val["taux"], "exactitude métier — traitement automatique": r_val["exactitude_metier_automatique"],
           "exactitude métier — demandes en revue": r_val["exactitude_metier_en_revue"]}).to_frame("validation")""",
    ),
    ("md", "## 6. Robustesse et équité"),
    ("code", """ev.robustesse(champion, validation)"""),
    (
        "md",
        """Le modèle résiste aux accents oubliés, aux majuscules (après correction de la casse des plongements : -15 points
avant), aux politesses et aux numéros de téléphone ajoutés. Les fautes de frappe denses restent le cas le plus
difficile, d'où l'entraînement sur des copies bruitées. **Équité** : le modèle n'utilise **que le texte** (ni adresse,
ni département, ni identité) ; on vérifie qu'aucun canal n'est nettement moins bien servi (les messages vocaux
transcrits et le téléphone sont souvent le fait de personnes moins à l'aise avec l'écrit) :""",
    ),
    (
        "code",
        """pd.concat([ev.par_tranche(champion, validation, "canal").assign(jeu="validation"),
           ev.par_tranche(champion, test_final, "canal").assign(jeu="test final")]).set_index(["jeu", "canal"])""",
    ),
    (
        "md",
        "## 7. Explicabilité\n\nPoids des termes les plus caractéristiques de chaque métier (partie TF-IDF « mots » de la régression logistique) :",
    ),
    (
        "code",
        """vect, clf = champion.modele_metier.named_steps["vect"], champion.modele_metier.named_steps["clf"]
noms = vect.get_feature_names_out(); mots = np.array([n.startswith("mots__") for n in noms])
pd.DataFrame({classe: [noms[j].removeprefix("mots__") for j in np.argsort(-np.where(mots, clf.coef_[i], -np.inf))[:8]]
              for i, classe in enumerate(clf.classes_)})""",
    ),
    (
        "code",
        """for texte in ["Ma fille de 6 ans est restée enfermée dans la salle de bain", "Le radiateur de la chambre gargouille et chauffe mal"]:
    p = champion.predire([texte], expliquer=True)[0]
    print(f"{texte}\\n  → {p.metier} ({p.proba_metier:.0%}), urgence {p.urgence}, revue humaine : {p.revue_humaine} {p.motifs_revue}")
    print("   termes :", ", ".join(f"{e['terme']} ({e['contribution']:+.2f})" for e in p.explication), "\\n")""",
    ),
    (
        "md",
        """La contribution « sens global de la phrase » (plongements) est souvent la plus forte : c'est elle qui permet de
reconnaître « ma fille est enfermée » sans mot-clé de serrurerie. L'API renvoie cette explication avec chaque prédiction.""",
    ),
    ("md", "## 8. Analyse d'erreurs (jeu de validation)"),
    (
        "code",
        """preds_val = champion.predire(validation["texte"].tolist())
erreurs = validation.assign(predit=[p.metier for p in preds_val], confiance=[p.proba_metier for p in preds_val],
                            revue=[p.revue_humaine for p in preds_val])
erreurs = erreurs[erreurs["predit"] != erreurs["metier"]]
print(f"{len(erreurs)} erreurs de métier sur {len(validation)} ; {erreurs['revue'].mean():.0%} d'entre elles partent en revue humaine")
erreurs[["texte", "metier", "predit", "confiance", "revue"]].sort_values("confiance", ascending=False)""",
    ),
    (
        "md",
        """Les erreurs restantes sont en majorité des **cas ambigus par nature** (ballon d'eau chaude : plomberie ou
chauffage ? radiateur électrique : électricité ou chauffage ? VMC) — un conseiller hésiterait aussi — et une bonne part
part déjà en revue humaine.

## 9. Limites et prochaines étapes

1. **Données synthétiques** : la vraie performance ne sera connue qu'en production, via le **taux de correction
   humaine** suivi en continu. Les corrections du support deviendront les nouvelles données d'entraînement.
2. **Petits jeux de test** : intervalles de confiance larges ; le test final a été mesuré plusieurs fois au fil des
   versions (déclaré) : la prochaine itération devra utiliser un **nouveau** jeu de test, idéalement de vraies demandes.
3. **Urgence** : la plus difficile (F1 ≈ 0,65) car elle dépend du contexte (« il gèle et ma mère est âgée ») ; d'où les
   garde-fous et la revue humaine. Piste : affiner un modèle français (CamemBERT) quand de vraies données seront disponibles.
4. **Coût** : plongements = ~80 ms par demande sur CPU (contre ~3 ms en TF-IDF seul), largement sous l'objectif de 300 ms.""",
    ),
]


def construire() -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t) for k, t in CELLULES]
    chemin = Path(__file__).parent / "01_exploration_modelisation.ipynb"
    nbf.write(nb, chemin)
    return chemin


if __name__ == "__main__":
    print(construire())
