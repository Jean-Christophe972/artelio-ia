"""Démonstration : une vraie dérive (vague de froid + nouveaux usages) détectée puis traitée.

    docker compose exec planificateur python scripts/demo_derive.py

Envoie 600 demandes à l'API qui ne ressemblent pas aux données d'entraînement :
  - vague de froid : les pannes de chauffage passent de ~7 % à ~60 % des demandes ;
  - nouveaux équipements absents du vocabulaire (pompe à chaleur hybride, borne de recharge,
    panneaux solaires, VMC double flux, domotique…) ;
  - messages vocaux transcrits, plus longs.
Puis lance l'analyse : Evidently doit détecter la dérive, et le planificateur déclenchera le
réentraînement au cycle suivant (raison « derive »).
"""

import json
import os
import random
import urllib.request

from monitoring.surveillance import analyser

rng = random.Random(1)
FROID = [
    "il fait 14 degrés dans le logement, la pompe à chaleur hybride affiche défaut compresseur",
    "le plancher chauffant ne chauffe plus et dehors il gèle, on a mis des couvertures partout",
    "chaudière à condensation en panne en plein hiver, plus de chauffage depuis vendredi soir",
    "les radiateurs restent glacés, le thermostat connecté indique erreur de communication",
    "le poêle à granulés s'éteint tout seul toutes les heures avec le froid qu'il fait",
]
NOUVEAUX = [
    "la borne de recharge du véhicule électrique ne délivre plus de courant depuis la mise à jour",
    "les panneaux solaires ne produisent plus rien et l'onduleur clignote en rouge",
    "la VMC double flux fait un bruit de turbine et le bypass reste bloqué",
    "la box domotique ne pilote plus les volets roulants ni les lumières connectées",
    "le ballon thermodynamique en mode boost tourne en continu sans chauffer l'eau",
]
VOCAL = (
    " euh alors voilà je vous appelle parce que bon c'est assez embêtant"
    " on a essayé de regarder nous-mêmes mais on ne sait pas trop quoi faire"
)

cle, url = os.environ["API_KEY"], os.environ.get("API_URL", "http://artelio-api:8000")
for _lot in range(12):  # 12 lots de 50 demandes
    lot = []
    for _ in range(50):
        texte = rng.choice(FROID) if rng.random() < 0.6 else rng.choice(NOUVEAUX)
        canal = rng.choice(["vocal", "telephone"]) if rng.random() < 0.5 else "app"
        lot.append({"texte": texte + (VOCAL if canal == "vocal" else ""), "canal": canal, "expliquer": False})
    req = urllib.request.Request(
        url + "/v1/qualifier/lot",
        data=json.dumps({"demandes": lot}).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": cle},
    )
    urllib.request.urlopen(req, timeout=60).read()
print("600 demandes « hivernales » envoyées à l'API ; analyse de dérive :")
print(analyser())
