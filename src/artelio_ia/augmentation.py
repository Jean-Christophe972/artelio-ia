"""Enrichissement des données d'entraînement (démarrage à froid).

Constat (modèle v1) : entraîné uniquement sur les demandes synthétiques du pipeline, le modèle
atteint F1 = 1,00 en validation croisée mais seulement 0,74 (métier) et 0,52 (urgence) sur des
textes écrits autrement : il a appris les GABARITS du générateur, pas le métier.

Correctif : on ajoute à l'entraînement des exemples construits à partir d'un LEXIQUE MÉTIER
(équipements, symptômes, signaux d'urgence), combinés selon plusieurs structures de phrase.
Ce lexique est écrit à partir de la connaissance du domaine, PAS à partir du jeu de test.
Quand Artelio disposera de vraies demandes corrigées par le support, leur part augmentera
et celle de l'enrichissement diminuera (paramètre n_par_combinaison).
"""

from __future__ import annotations

import random
import unicodedata

import pandas as pd

from artelio_ia.config import GRAINE

# Équipements / objets typiques de chaque métier
OBJETS = {
    "plomberie": [
        "évier",
        "lavabo",
        "WC",
        "toilettes",
        "chasse d'eau",
        "siphon",
        "tuyau",
        "canalisation",
        "flexible",
        "robinet",
        "mitigeur",
        "douche",
        "baignoire",
        "arrivée d'eau",
        "évacuation",
        "compteur d'eau",
        "bonde",
    ],
    "electricite": [
        "prise",
        "disjoncteur",
        "tableau électrique",
        "interrupteur",
        "câble",
        "fusible",
        "différentiel",
        "luminaire",
        "plafonnier",
        "circuit",
        "compteur électrique",
        "interphone",
        "VMC",
        "spots",
    ],
    "serrurerie": [
        "serrure",
        "porte d'entrée",
        "cylindre",
        "barillet",
        "clé",
        "verrou",
        "poignée",
        "porte blindée",
        "serrure trois points",
        "gâche",
        "boîte aux lettres",
        "digicode",
    ],
    "chauffage": [
        "chaudière",
        "radiateur",
        "ballon d'eau chaude",
        "cumulus",
        "chauffe-eau",
        "pompe à chaleur",
        "thermostat",
        "climatisation",
        "plancher chauffant",
        "brûleur",
        "poêle à granulés",
    ],
    "peinture": [
        "mur",
        "plafond",
        "peinture",
        "papier peint",
        "enduit",
        "façade",
        "boiseries",
        "volets",
        "crépi",
        "toile de verre",
        "cage d'escalier",
    ],
    "menuiserie": [
        "fenêtre",
        "porte-fenêtre",
        "baie vitrée",
        "vitre",
        "volet roulant",
        "placard",
        "parquet",
        "escalier",
        "rambarde",
        "porte de chambre",
        "velux",
        "tiroir",
        "plinthes",
    ],
    "petits_travaux": [
        "étagère",
        "meuble en kit",
        "tringle à rideaux",
        "support télé",
        "miroir",
        "cadre",
        "joints",
        "barre d'appui",
        "store",
        "patère",
        "lit mezzanine",
        "crochets",
    ],
}

# Symptômes par niveau d'urgence (politique : critique = danger ou dégâts en cours,
# haute = service essentiel indisponible, normale = gêne, faible = travaux planifiables)
SYMPTOMES = {
    "plomberie": {
        "critique": [
            "fuit énormément et inonde la pièce",
            "a éclaté, l'eau jaillit",
            "provoque un dégât des eaux chez le voisin",
            "déborde et l'eau se répand partout",
            "fuit et je n'arrive pas à couper l'eau",
        ],
        "haute": ["est complètement bouché", "ne s'évacue plus du tout", "n'a plus d'eau", "refoule des eaux usées", "fuit en continu"],
        "normale": [
            "goutte sans arrêt",
            "fuit un peu",
            "fait un bruit bizarre",
            "s'écoule lentement",
            "a une mauvaise odeur",
            "manque de pression",
        ],
        "faible": ["est à remplacer", "est à installer", "doit être changé pour un modèle neuf", "est à déplacer pendant les travaux"],
    },
    "electricite": {
        "critique": [
            "fait des étincelles et sent le brûlé",
            "a pris feu",
            "dégage de la fumée",
            "a des fils dénudés qui pendent",
            "est noirci et sent le cramé",
        ],
        "haute": [
            "disjoncte sans arrêt",
            "a coupé le courant dans toute la cuisine",
            "ne fonctionne plus, plus de courant",
            "saute dès qu'on branche un appareil",
        ],
        "normale": ["ne fonctionne plus", "grésille", "clignote", "fait du bruit", "marche une fois sur deux"],
        "faible": ["est à installer", "est à ajouter", "doit être mis aux normes", "est à remplacer par un modèle récent"],
    },
    "serrurerie": {
        "critique": [
            "a claqué et je suis bloqué dehors",
            "a été forcée lors d'un cambriolage",
            "est fracturée et ne ferme plus",
            "s'est fermée avec mon enfant à l'intérieur",
            "a été arrachée cette nuit",
        ],
        "haute": ["est cassée dans la serrure", "ne se verrouille plus", "tourne dans le vide", "ne ferme plus à clé"],
        "normale": ["est grippée", "frotte et ferme mal", "est dure à tourner", "accroche"],
        "faible": ["est à changer", "est à installer", "est à doubler par sécurité", "est à refaire en double"],
    },
    "chauffage": {
        "critique": [
            "dégage une odeur de gaz",
            "fait sonner le détecteur de monoxyde",
            "est en panne alors qu'il gèle et qu'un bébé est là",
            "fuit du gaz",
            "est en panne chez une personne âgée sans chauffage par grand froid",
        ],
        "haute": [
            "est en panne, plus de chauffage",
            "ne fait plus d'eau chaude",
            "affiche un code erreur et s'est arrêtée",
            "reste froid malgré le thermostat à fond",
        ],
        "normale": ["chauffe mal", "fait du bruit", "gargouille", "se coupe de temps en temps", "doit être purgé"],
        "faible": [
            "a besoin de son entretien annuel",
            "est à remplacer par un modèle plus économe",
            "est à installer",
            "doit être désembouée avant l'hiver",
        ],
    },
    "peinture": {
        "critique": [],  # pas d'urgence critique en peinture
        "haute": [],
        "normale": [
            "a des traces de moisissure",
            "est taché et cloqué après un dégât des eaux",
            "est fissuré",
            "s'écaille et gonfle avec l'humidité",
        ],
        "faible": ["est à repeindre", "est à rafraîchir", "doit être refait en blanc", "est à poser", "est à lessiver et repeindre"],
    },
    "menuiserie": {
        "critique": [],
        "haute": [
            "est cassée, il y a des éclats de verre",
            "ne ferme plus et il va pleuvoir",
            "est arraché par le vent",
            "bouge et c'est dangereux pour les enfants",
        ],
        "normale": ["est bloqué", "est sorti de son rail", "est fêlé", "s'est soulevé", "ferme mal", "est déformé"],
        "faible": ["est à poser", "est à fabriquer sur mesure", "est à remplacer par du double vitrage", "est à ajuster"],
    },
    "petits_travaux": {
        "critique": [],
        "haute": [],
        "normale": [
            "est à poser vite pour une personne à mobilité réduite",
            "est arraché et à refixer",
            "est à installer avant l'arrivée des locataires",
        ],
        "faible": ["est à monter", "est à fixer au mur", "est à accrocher", "est à poser", "est à assembler"],
    },
}

STRUCTURES = [
    "{debut}{le} {objet} {symptome}{lieu}.{fin}",
    "{debut}problème de {objet} : il {symptome}{lieu}.{fin}",
    "{debut}chez moi {le} {objet} {symptome}, pouvez-vous intervenir ?{fin}",
    "{debut}{objet} qui {symptome}{lieu}{fin}",
    "{debut}j'ai besoin d'un artisan, {le} {objet} {symptome}.{fin}",
]
DEBUTS = ["", "", "", "Bonjour, ", "Bonsoir, ", "Bonjour madame, monsieur, ", "Salut ", "euh ", "Allô, "]
FINS = ["", "", "", " Merci.", " Merci de me rappeler.", " C'est assez pressé.", " Je suis disponible ce soir.", " Combien ça coûte ?"]
LIEUX = ["", "", " dans la cuisine", " dans la salle de bain", " au salon", " dans la chambre", " à l'entrée", " au garage"]
# Signaux d'urgence ajoutés uniquement aux exemples critiques (le niveau d'urgence ne dépend pas d'un seul mot)
APPUIS_CRITIQUES = [
    "",
    "",
    " Au secours !",
    " C'est très urgent, c'est dangereux.",
    " Venez au plus vite svp",
    " Il est minuit et je ne sais pas quoi faire.",
]


def _sans_accents(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")


def _bruit(t: str, rng: random.Random) -> str:
    """Saisie mobile : accents oubliés, minuscules, lettres inversées."""
    if rng.random() < 0.3:
        t = _sans_accents(t)
    if rng.random() < 0.3:
        t = t.lower()
    if rng.random() < 0.2 and len(t) > 15:
        i = rng.randrange(3, len(t) - 2)
        t = t[:i] + t[i + 1] + t[i] + t[i + 2 :]
    return t


def generer(n_par_combinaison: int = 120, graine: int = GRAINE) -> pd.DataFrame:
    """Exemples étiquetés (métier, urgence), équilibrés entre combinaisons existantes."""
    rng = random.Random(graine)
    lignes = []
    for metier, par_urgence in SYMPTOMES.items():
        for urgence, symptomes in par_urgence.items():
            for _ in range(n_par_combinaison if symptomes else 0):
                objet = rng.choice(OBJETS[metier])
                texte = rng.choice(STRUCTURES).format(
                    debut=rng.choice(DEBUTS),
                    le="l'" if objet[0].lower() in "aeiouéèh" else "le",
                    objet=objet,
                    symptome=rng.choice(symptomes),
                    lieu=rng.choice(LIEUX),
                    fin=rng.choice(APPUIS_CRITIQUES if urgence == "critique" else FINS),
                )
                lignes.append(
                    {"texte": _bruit(texte[0].upper() + texte[1:], rng), "metier": metier, "urgence": urgence, "source": "lexique"}
                )
    return pd.DataFrame(lignes)


def copies_bruitees(donnees: pd.DataFrame, part: float = 0.3, taux: float = 0.05, graine: int = GRAINE) -> pd.DataFrame:
    """Copies d'une partie des exemples avec des fautes de frappe (lettres voisines inversées).

    Constat : 6 % de lettres inversées faisaient perdre 11 points d'exactitude au modèle. Apprendre
    sur des textes abîmés, comme ceux tapés vite sur mobile, le rend robuste à ces fautes.
    """
    rng = random.Random(graine)
    echantillon = donnees.sample(frac=part, random_state=graine)

    def fautes(texte: str) -> str:
        lettres = list(texte)
        for i in range(len(lettres) - 1):
            if lettres[i].isalpha() and rng.random() < taux:
                lettres[i], lettres[i + 1] = lettres[i + 1], lettres[i]
        return "".join(lettres)

    return echantillon.assign(texte=echantillon["texte"].map(fautes), source="bruit")
