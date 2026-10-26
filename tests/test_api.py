"""API : authentification, validation des entrées, contrat de réponse, métriques."""

import pytest
from conftest import CLE_API
from fastapi.testclient import TestClient

from artelio_ia import registre


@pytest.fixture
def client(qualifieur, tmp_path, monkeypatch, empreinte_cle):
    registre.sauvegarder(qualifieur, tmp_path)
    monkeypatch.setenv("MODELE_SOURCE", "local")
    monkeypatch.setenv("MODELE_DOSSIER", str(tmp_path))
    monkeypatch.delenv("PG_HOST", raising=False)
    import api.main as module

    monkeypatch.setattr(module, "EMPREINTES", [empreinte_cle])
    module.modele.source = "local"
    with TestClient(module.app) as c:
        yield c


def test_sante_et_pret(client):
    assert client.get("/sante").json() == {"statut": "ok"}
    assert client.get("/pret").json()["statut"] == "pret"


def test_cle_obligatoire(client):
    assert client.post("/v1/qualifier", json={"texte": "fuite sous l'évier"}).status_code == 401
    assert client.post("/v1/qualifier", json={"texte": "fuite"}, headers={"X-API-Key": "mauvaise"}).status_code == 401


def test_qualification(client):
    r = client.post("/v1/qualifier", json={"texte": "Le robinet de la cuisine goutte", "canal": "web"}, headers={"X-API-Key": CLE_API})
    assert r.status_code == 200
    corps = r.json()
    assert corps["metier"] == "plomberie"
    assert set(corps) >= {"urgence", "confiance_metier", "revue_humaine", "explication", "version_modele"}


@pytest.mark.parametrize(
    "corps",
    [{"texte": "a"}, {"texte": "x" * 2001}, {"texte": "fuite d'eau", "canal": "pigeon"}, {"texte": "fuite d'eau", "inconnu": 1}],
)
def test_entrees_invalides(client, corps):
    assert client.post("/v1/qualifier", json=corps, headers={"X-API-Key": CLE_API}).status_code == 422


def test_lot_limite_a_100(client):
    lot = {"demandes": [{"texte": "fuite d'eau"}] * 101}
    assert client.post("/v1/qualifier/lot", json=lot, headers={"X-API-Key": CLE_API}).status_code == 422


def test_metriques_exposees(client):
    client.post("/v1/qualifier", json={"texte": "serrure bloquée"}, headers={"X-API-Key": CLE_API})
    assert "artelio_ia_predictions_total" in client.get("/metrics").text
