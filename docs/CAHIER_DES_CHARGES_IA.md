# Cahier des charges — Qualification des demandes par IA

## 1. Problème métier

Chaque jour, ~330 demandes arrivent (10 000 / mois, ×2 en hiver). Aujourd'hui, un conseiller lit chaque message pour
choisir le métier et juger l'urgence : 2 à 3 minutes par demande, délais allongés le soir et le week-end, urgences
parfois traitées trop tard. **Objectif** : pré-qualifier automatiquement chaque demande en moins d'une seconde, pour
contacter plus vite le bon artisan et traiter d'abord ce qui est dangereux.

## 2. Exigences fonctionnelles

| # | Exigence | Réponse |
|---|---|---|
| F1 | Prédire le métier parmi 7 | Classifieur métier |
| F2 | Prédire l'urgence (faible, normale, haute, critique) | Classifieur urgence + garde-fous |
| F3 | Indiquer la confiance et demander une validation humaine en cas de doute | Probabilités, revue humaine (confiance < 0,55, texte hors domaine, urgence critique suspectée) |
| F4 | Expliquer la décision | Termes déterminants + « sens global de la phrase » dans chaque réponse |
| F5 | S'intégrer au flux existant | Service de qualification : demandes du pipeline (bloc 3) → API → base (bloc 2) |
| F6 | Traiter des lots | `POST /v1/qualifier/lot` (100 demandes) |

## 3. Exigences de performance (critères d'acceptation)

| # | Critère | Seuil | Mesuré (champion) |
|---|---|---|---|
| P1 | F1 macro métier sur textes hors gabarits (validation) | ≥ 0,70 | 0,91 |
| P2 | Rappel des urgences critiques (validation) | ≥ 0,85 | 1,00 |
| P3 | Urgences critiques classées « normale » ou « faible » (test final) | ≤ 2 % | 1,1 % |
| P4 | Latence p95 de l'API | < 300 ms | ~160 ms en médiane (bout en bout) |
| P5 | Non-régression lors d'un réentraînement | F1 ≥ champion − 1 point sur les mêmes jeux | barrière de promotion automatique |
| P6 | Robustesse (fautes, accents, majuscules, coordonnées ajoutées) | perte ≤ 10 points (12 pour les fautes denses, limite documentée) | voir fiche modèle |

## 4. Exigences d'accessibilité et d'inclusion

| Exigence | Réponse |
|---|---|
| Les demandes passées par **téléphone** ou **message vocal** (personnes malvoyantes, peu à l'aise avec l'écrit) ne doivent pas être moins bien comprises | Performance suivie **par canal** (notebook §6, surveillance) ; modèle entraîné sur du style oral et des transcriptions |
| Les fautes, l'absence d'accents et le style SMS (dyslexie, non-francophones) ne doivent pas pénaliser | Variables par n-grammes de caractères, normalisation de la casse et des accents, tests de robustesse dans la CI |
| Une personne vulnérable (âgée, enfant, mobilité réduite) doit être priorisée | Garde-fous « froid + personne vulnérable », « enfant enfermé » |
| Les réponses de l'IA doivent être compréhensibles par un conseiller non technicien | Libellés métier en clair, motifs de revue explicites, explication par termes |

## 5. Exigences d'exploitation (MLOps)

| # | Exigence | Réponse |
|---|---|---|
| O1 | Déploiement automatisé et sans coupure | CI/CD GitHub Actions → image Docker → Railway ; rechargement à chaud du champion ; manifestes Kubernetes (`maxUnavailable: 0`) |
| O2 | Réentraînement automatique | Nouvelle version des données, dérive détectée ou modèle de plus de 7 jours |
| O3 | Surveillance de la dérive et de la performance réelle | Evidently (PSI, Wasserstein) chaque heure ; taux de correction humaine ; Prometheus + Grafana + alertes |
| O4 | Reproductibilité et traçabilité | Données versionnées + SHA-256, graine fixe, dépendances épinglées, MLflow (paramètres, métriques, artefacts, registre) |
| O5 | Sécurité | Clé d'API (empreinte seule côté serveur), limitation de débit, validation des entrées, artefact vérifié par SHA-256 avant chargement, utilisateur non root, réseau interne, secrets hors du code |

## 6. Conformité

| Texte | Exigence | Réponse |
|---|---|---|
| RGPD art. 5, 25 | Minimisation, protection dès la conception | Le modèle ne reçoit que le texte ; textes masqués avant journalisation ; jeu d'entraînement pseudonymisé |
| RGPD art. 21 | Droit d'opposition à l'entraînement | Clients opposés exclus du jeu d'entraînement (bloc 3) |
| RGPD art. 22 | Pas de décision entièrement automatisée défavorable | Aide à la décision ; le client choisit l'artisan ; revue humaine |
| Loi Informatique et Libertés | Droits des personnes, registre | Journal des prédictions effacé à 90 jours ; traitement inscrit au registre (bloc 1) |
| AI Act art. 50 | Transparence | Champ `avertissement` dans chaque réponse : « pré-qualification automatique par IA » |
| AI Act art. 4 | Maîtrise de l'IA par le personnel | Formation des conseillers à la lecture des probabilités et des motifs de revue (plan de gouvernance §10.2) |
| ISO 27001 | Sécurité | Voir O5 ; journalisation des accès ; analyse de vulnérabilités de l'image (Trivy) dans la CI |
