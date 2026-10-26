# Architecture MLOps

## 1. Cycle de vie du modèle

```mermaid
flowchart TB
    subgraph donnees["Données (bloc 3)"]
        DS["Jeu versionné S3<br/>+ manifeste SHA-256"]
        CAT[("pipeline.datasets<br/>empreinte de référence")]
    end
    subgraph entrainement["Entraînement (image ml)"]
        V["Vérification d'intégrité"] --> SPLIT["Découpage temporel<br/>+ enrichissement + bruit"]
        SPLIT --> SEL["Sélection (manuel) :<br/>9 candidats, choix sur la validation hors gabarits<br/>— ou —<br/>Réentraînement (auto) :<br/>configuration du champion"]
        SEL --> EV["Évaluation : test temporel, validation,<br/>test final, robustesse, canaux, calibration, latence"]
        EV --> GATE{"Barrière de promotion :<br/>planchers métier +<br/>pas de régression vs champion"}
    end
    DS --> V
    CAT --> V
    GATE -->|oui| CH["MLflow : alias « champion »"]
    GATE -->|non| CL["MLflow : alias « challenger »<br/>+ raison du refus"]
    CH -->|rechargement à chaud (60 s)| API["API"]
    API --> LOG[("ml.predictions")]
    LOG --> MON["Surveillance Evidently (1 h)<br/>+ taux de correction humaine"]
    MON -->|dérive| TRIG["Planificateur"]
    DS -->|nouvelle version| TRIG
    TRIG --> V
```

## 2. Composants

| Composant | Rôle | Technologie |
|---|---|---|
| Image `ml` | Entraînement, notebook, surveillance, planificateur, service de qualification | Python 3.12, scikit-learn 1.6, sentence-transformers 3.4 (MiniLM figé dans l'image), MLflow 2.22, Evidently 0.4 |
| Image `api` | Prédiction en ligne | FastAPI, Uvicorn, utilisateur non root, modèle de plongements inclus (aucun téléchargement à l'exécution) |
| MLflow | Suivi des expériences (paramètres, métriques, tableaux, matrices de confusion) et **registre** (versions, alias) | Base SQLite (volume) + artefacts dans le bucket S3 `mlflow`, servis par le serveur (les clients n'ont pas les clés S3) |
| Service de qualification | Toutes les 5 s : demandes « nouvelle » → API en lot → base | Compte `ml_service` (colonnes utiles uniquement) |
| Planificateur | Surveillance horaire ; réentraînement sur nouvelle version / dérive / âge > 7 jours | Processus Python, décisions fondées sur la **dernière tentative** (pas de boucle si un modèle est refusé) |
| Supervision | Métriques de l'API, de l'entraînement, de la dérive ; 5 alertes IA | Prometheus, Pushgateway, Alertmanager, Grafana (plateforme du bloc 2) |

## 3. CI/CD

| Étape (GitHub Actions) | Contrôle | Bloquant |
|---|---|---|
| Analyse statique | ruff (PEP 8, imports, bugs courants), format | oui |
| Tests | 42 tests unitaires et d'API | oui |
| Barrière de qualité du modèle livré | Planchers sur la validation, sous-estimation des urgences sur le test final, robustesse | oui |
| Image Docker | Construction, **test de fumée** (l'image démarre et qualifie une demande), analyse de vulnérabilités **Trivy** (CRITICAL) | oui |
| Publication | Image poussée sur GHCR (`latest` + SHA du commit) — branche `main` uniquement | — |
| Déploiement (CD) | Après CI réussie sur `main` : `railway up`, puis vérification de `/pret` en production | oui |

**Deux chemins de mise en production complémentaires :**
- **Code** (nouvelle fonctionnalité, correctif) : commit → CI → image → Railway. L'image embarque `models/champion`, un
  modèle validé par la barrière de qualité : le déploiement est immuable et reproductible.
- **Modèle** (nouvelles données, dérive) : réentraînement → barrière de promotion → alias `champion` → l'API de la
  plateforme le charge à chaud, sans redéploiement. Le champion est exporté dans `models/champion` pour la prochaine image.

## 4. Surveillance

| Signal | Source | Seuil d'alerte |
|---|---|---|
| Disponibilité, erreurs 5xx, latence p95 | API (`/metrics`) | indisponible 1 min ; > 1 % ; > 300 ms |
| Part en revue humaine | API | > 30 % pendant 30 min |
| Dérive des données (8 variables : longueur, mots, mots inconnus, confiances, classes prédites, canal) | Evidently, chaque heure, 1 000 dernières prédictions vs référence | > 30 % des variables en dérive |
| Performance réelle | Part des métiers prédits corrigés par le support (7 jours) | > 15 % |

**Choix du test de dérive.** Sur des milliers de prédictions, un test de significativité (khi-deux, Kolmogorov-Smirnov)
déclare « dérive » pour des écarts insignifiants : constaté sur la plateforme (55,3 % contre 53,6 % de demandes via
l'application → p ≈ 0). On mesure donc l'**ampleur** de l'écart : PSI pour les variables catégorielles (seuil 0,2),
distance de Wasserstein normalisée pour les numériques (seuil 0,1).

## 5. Réentraînement : ce qui a été démontré

| Événement | Résultat |
|---|---|
| 600 demandes « vague de froid + nouveaux équipements » | Dérive détectée (confiance, mots inconnus, longueur) → réentraînement automatique → v2 promue → API passée de v1 à v2 sans coupure |
| Entraînement avec copies bruitées (v4) | **Refusé** par la barrière (urgence −1,3 point sur la validation) → reste `challenger`, le champion reste en service |

## 6. Sécurité

- Clé d'API comparée à temps constant à son **empreinte** SHA-256 ; aucune clé en clair côté serveur ; échec sûr si
  aucune clé n'est configurée ; limitation de débit (20 req/s, rafales 40) ; erreurs internes jamais renvoyées au client.
- Entrées validées (longueur, canal, champs inconnus refusés) ; lots limités à 100.
- Artefacts : empreinte SHA-256 vérifiée avant `joblib.load` (un pickle peut exécuter du code).
- Données d'entraînement : empreinte vérifiée contre le catalogue PostgreSQL, que le service IA ne peut pas modifier.
- Conteneurs non root, réseau interne sans accès Internet, secrets uniquement dans `.env` (non versionné) ou le
  gestionnaire de secrets de la plateforme cible.
