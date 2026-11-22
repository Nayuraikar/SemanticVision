import numpy as np
import pytest
from fastapi.testclient import TestClient

from calibration.model import SimObserver
from core.server import create_app

CFG = {"seed": 1, "min_trials": 6, "max_trials": 8, "n_particles": 200}


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(gt_dir=tmp_path / "gt", profiles_db=tmp_path / "p.sqlite"))


def _finish(client, seed):
    pid = client.post("/profiles", json={"config": {**CFG, "seed": seed}}).json()["id"]
    truth, rng = SimObserver("machado", "deutan", 0.9), np.random.default_rng(0)
    while True:
        j = client.get(f"/profiles/{pid}/next").json()
        if j["done"]:
            break
        st = j["stimulus"]
        client.post(f"/profiles/{pid}/answer", json={"trial_id": st["trial_id"], "response": truth.respond(
            st["mode"], st["c1"], st["c2"], st["size_deg"], st["bg_lum"], rng)})
    assert client.post(f"/profiles/{pid}/finish").status_code == 200
    return pid


def test_get_profiles_lists_only_complete(client):
    assert client.get("/profiles").json() == []
    open_pid = client.post("/profiles", json={"config": CFG}).json()["id"]
    assert client.get("/profiles").json() == []
    done = _finish(client, 2)
    lst = client.get("/profiles").json()
    assert [p["id"] for p in lst] == [done] and open_pid not in [p["id"] for p in lst]
    p = lst[0]
    assert p["observer_id"] == f"profile:{done}" and p["status"] == "complete"
    assert p["kind"] in ("protan", "deutan", "tritan") and 0 <= p["severity"] <= 1
    second = _finish(client, 3)
    assert {x["id"] for x in client.get("/profiles").json()} == {done, second}
