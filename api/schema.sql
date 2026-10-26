-- Journal des prédictions de l'IA (schéma "ml", propriété du compte ml_service).
-- Texte masqué (pas de téléphone, email, IBAN) ; conservation 90 jours (purge par la surveillance).
CREATE TABLE IF NOT EXISTS ml.predictions (
    id                     bigserial PRIMARY KEY,
    horodatage             timestamptz NOT NULL DEFAULT now(),
    demande_id             uuid,
    version_modele         text NOT NULL,
    texte_masque           text NOT NULL,
    longueur               integer NOT NULL,
    nb_mots                integer NOT NULL,
    taux_hors_vocabulaire  real NOT NULL,
    canal                  text NOT NULL,
    metier                 text NOT NULL,
    proba_metier           real NOT NULL,
    urgence                text NOT NULL,
    proba_urgence          real NOT NULL,
    revue_humaine          boolean NOT NULL,
    motifs                 text[] NOT NULL DEFAULT '{}',
    latence_ms             real NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_horodatage ON ml.predictions (horodatage);

CREATE TABLE IF NOT EXISTS ml.rapports_derive (
    id                 bigserial PRIMARY KEY,
    horodatage         timestamptz NOT NULL DEFAULT now(),
    version_modele     text,
    n_reference        integer NOT NULL,
    n_courant          integer NOT NULL,
    derive_detectee    boolean NOT NULL,
    part_variables     real NOT NULL,
    variables_derivees text[] NOT NULL,
    taux_correction    real,
    cle_rapport        text
);

-- Le DPO et les analystes peuvent auditer les décisions de l'IA
GRANT SELECT ON ml.predictions, ml.rapports_derive TO dpo_officer, analyst;
