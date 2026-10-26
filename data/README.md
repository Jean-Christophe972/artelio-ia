# Données du modèle de qualification

## Origine des données d'entraînement

| Source | Volume | Rôle |
|---|---|---|
| Jeu publié par le pipeline (bloc 3), bucket S3 `datasets`, versionné | ~7 400 demandes | Données « de production » (synthétiques), étiquetées métier + urgence |
| Enrichissement par lexique métier (`src/artelio_ia/augmentation.py`) | ~2 800 exemples | Diversité des formulations (démarrage à froid) |
| Copies bruitées (fautes de frappe) | 30 % des exemples | Robustesse aux saisies mobiles |

Le jeu du pipeline est **pseudonymisé** (identifiants HMAC), **masqué** (téléphones, emails), et exclut les clients
qui se sont **opposés** à l'entraînement de l'IA. Son empreinte SHA-256 est vérifiée contre le catalogue
`pipeline.datasets` (PostgreSQL) avant chaque entraînement.

## Les trois jeux d'évaluation

| Jeu | Contenu | Usage | Risque de biais |
|---|---|---|---|
| **Test temporel** | Les 60 derniers jours du jeu du pipeline | Mesure sur le « futur » | Élevé : mêmes gabarits que l'entraînement → toujours ~100 %, **non représentatif** |
| **Validation hors gabarits** (`validation_hors_gabarits.csv`, 143 textes) | Écrits à la main, formulations variées | **Sélection de modèle et barrière de promotion** | Moyen : a guidé le développement (lexique, garde-fous) |
| **Test final hors gabarits** (`test_hors_gabarits.csv`, 90 textes) | Écrits **après le gel de la v2**, nouveau vocabulaire (Linky, WC broyeur, style SMS…) | **Mesure uniquement**, jamais utilisé pour décider | Le plus faible ; rédigé par l'équipe (pas par de vrais clients) |

### Pourquoi c'est important

Les données du pipeline sont générées à partir de gabarits de phrases. Un modèle y obtient **F1 = 1,00 en validation
croisée**, mais seulement **0,74** sur des textes formulés autrement (modèle v1) : il avait appris les gabarits, pas
le métier. Toute l'évaluation repose donc sur des textes hors gabarits, et le chiffre de référence est celui du
**test final**.

### Politique d'étiquetage de l'urgence

| Niveau | Définition | Exemples |
|---|---|---|
| critique | Danger pour les personnes ou dégâts en cours | gaz, étincelles, inondation, enfant enfermé, effraction, personne vulnérable sans chauffage par grand froid |
| haute | Service essentiel indisponible, sans danger immédiat | plus de chauffage ni d'eau chaude, WC bouchés, clé cassée, vitre brisée |
| normale | Gêne, à traiter dans la semaine | robinet qui goutte, volet coincé, moisissures |
| faible | Travaux planifiables | peinture, installation, montage de meubles, entretien |

Les textes ambigus sont conservés volontairement (ex. « ballon d'eau chaude qui fuit » : plomberie ou chauffage ?).

### Limites

- Les jeux hors gabarits sont rédigés par l'équipe, pas par de vrais clients : ils restent une approximation.
- Petits effectifs (90 textes, dont 14 urgences critiques dans le test final) : **un cas = 7 points de rappel** ;
  les intervalles de confiance (bootstrap) sont donnés dans le notebook et la fiche modèle.
- La mesure réelle viendra de la production : le **taux de correction humaine** (métiers corrigés par le support)
  est suivi en continu par la surveillance (`monitoring/surveillance.py`).
