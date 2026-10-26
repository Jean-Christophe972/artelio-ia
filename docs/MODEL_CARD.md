# Fiche modèle — `artelio-qualifieur`

> Format inspiré des *Model Cards* (Mitchell et al., 2019). Version décrite : **champion v3** du registre MLflow
> (exporté dans `models/champion/`). Détail des calculs : `notebooks/01_exploration_modelisation.ipynb`.

## 1. Usage prévu

| | |
|---|---|
| **Tâche** | À partir du texte libre d'une demande de dépannage, prédire le **métier** (7 classes) et l'**urgence** (4 niveaux) |
| **Utilisateurs** | Le service de qualification d'Artelio (automatique) ; les conseillers du support (revue humaine) |
| **Décision** | **Aide à la décision** : le modèle pré-qualifie, un humain valide quand le modèle doute ; aucune décision défavorable à une personne |
| **Hors périmètre** | Évaluer un artisan, fixer un prix, décider seul d'une intervention d'urgence, traiter des textes non français |
| **Niveau de risque (AI Act)** | Limité (art. 50, transparence) : le client est informé que sa demande est pré-analysée par une IA |

## 2. Modèle

| Élément | Choix | Justification |
|---|---|---|
| Variables | TF-IDF mots (1-2) + TF-IDF caractères (2-5, robustesse aux fautes) + **plongements de phrases** MiniLM multilingue (384 dim.) | Vocabulaire du métier + sens de la phrase sans mot commun |
| Classifieur | Régression logistique (`class_weight="balanced"`), un modèle par tâche | Probabilités exploitables, poids lisibles, prédiction CPU |
| Hyperparamètres | Métier : C = 1 ; urgence : C = 4 ; poids des plongements 1,5 — choisis sur la validation hors gabarits | — |
| Politique de décision | Revue humaine si confiance métier < 0,55 ou texte hors domaine (≥ 50 % de mots inconnus) ; urgence forcée à « critique » si P(critique) ≥ 0,25 ou signal de danger explicite (garde-fous) | Coût asymétrique : une urgence ratée coûte bien plus qu'une fausse alerte |
| Alternatives évaluées | Classe majoritaire, Naive Bayes, SVM linéaire, régression logistique TF-IDF (C = 1, 4, 16), plongements seuls, hybrides (C = 1, 4) | Tableau complet dans MLflow (`selection/validation_croisee_*.json`) |
| Alternative écartée | Affinage de CamemBERT | Pas de vraies données étiquetées ; GPU ; latence ; explicabilité |

## 3. Données

- **Entraînement** : jeu versionné publié par le pipeline (bloc 3), ~7 400 demandes (5 800 après découpage temporel),
  + 2 760 exemples d'un lexique métier + copies bruitées. Empreinte SHA-256 vérifiée contre le catalogue PostgreSQL.
- **Pseudonymisé, masqué**, sans les clients qui se sont opposés à l'entraînement (212 demandes exclues).
- Détail des trois jeux d'évaluation et de leurs biais : [`data/README.md`](../data/README.md).

## 4. Performances

### Sur des textes jamais vus (test final, 90 textes)

| Métrique | Valeur | IC 95 % (bootstrap) |
|---|---|---|
| F1 macro métier | **0,78** | 0,68 – 0,86 |
| F1 macro urgence | **0,64** | 0,53 – 0,74 |
| Rappel des urgences critiques | **0,79** | 0,55 – 1,00 |
| Urgences sous-estimées de 2 niveaux ou plus | 1,1 % | — |
| Part en revue humaine | 38 % (exactitude métier 87,5 % sur la part traitée automatiquement) | — |
| ECE (calibration métier) | 0,05 | — |

### Sur la validation hors gabarits (143 textes, a servi aux choix)

| F1 métier | F1 urgence | Rappel critique | Revue humaine | Exactitude si traitement automatique | ECE (calibration) |
|---|---|---|---|---|---|
| 0,91 | 0,72 | 1,00 | 20 % | 94 % | 0,08 |

### À ne pas confondre

Validation croisée et test temporel donnent ~1,00 : ils sont construits avec les mêmes gabarits que l'entraînement et
**ne mesurent pas** la généralisation. Le modèle v1 (TF-IDF seul) avait 1,00 en validation croisée et 0,74 hors gabarits.

### Évolution

| Version | F1 métier (test) | F1 urgence (test) | Rappel critique (test) | Décision |
|---|---|---|---|---|
| v2 — TF-IDF + lexique + garde-fous | 0,80 | 0,59 | 0,57 | remplacée |
| v3 — hybride + casse + hors domaine | 0,78 | 0,64 | 0,79 | **champion** |
| v4 — + copies bruitées | 0,79 | 0,63 | 0,79 | **refusée** par la barrière (urgence −1,3 pt en validation) |

## 5. Robustesse, équité, explicabilité

- **Robustesse** (perte d'exactitude métier) : accents oubliés −3,5 pts ; majuscules 0 ; numéro de téléphone ajouté −3,5 ;
  formules de politesse −2,8 ; **fautes de frappe denses −11,2 (limite connue)** ; texte tronqué à 5 mots −12,6.
- **Équité** : le modèle n'utilise que le texte (ni adresse, ni identité). Performance suivie par canal : sur le test final,
  le canal *app* est le moins bien servi (exactitude métier 0,68 sur 31 textes, plus courts) → davantage de revue humaine.
- **Explicabilité** : chaque réponse de l'API liste les termes qui ont pesé dans le choix du métier, plus la
  contribution du « sens global de la phrase ».

## 6. Limites et risques

| Limite | Mesure de maîtrise |
|---|---|
| Données d'entraînement synthétiques | Suivi en production du **taux de correction humaine** ; les corrections deviennent les données de demain |
| Petits jeux de test, test final mesuré plusieurs fois | Intervalles de confiance publiés ; **nouveau jeu de test** à constituer avec de vraies demandes |
| Urgence dépendante du contexte (« il gèle et ma mère est âgée ») | Garde-fous, seuil de précaution, revue humaine |
| Textes ambigus (ballon d'eau chaude, VMC, radiateur électrique) | Probabilités par classe renvoyées ; revue humaine si confiance faible |
| Surconfiance hors domaine | Détection « hors domaine » par la part de mots inconnus |
| Dérive saisonnière (chauffage l'hiver) et nouveaux équipements | Surveillance Evidently horaire, réentraînement automatique avec barrière de promotion |

## 7. Exploitation

| | |
|---|---|
| Latence | 80 à 240 ms par prédiction selon la charge de la machine (CPU, mesuré pendant l'entraînement) ; médiane de 160 ms de bout en bout via l'API ; objectif < 300 ms |
| Reproductibilité | Version des données + SHA-256, graine fixe, versions de bibliothèques épinglées, commit Git, tout enregistré dans MLflow |
| Réentraînement | Nouvelle version des données, dérive détectée ou modèle de plus de 7 jours → même configuration, nouvelles données, barrière de promotion |
| Journal des prédictions | `ml.predictions` (texte masqué, 90 jours) : dérive, audit des décisions |
