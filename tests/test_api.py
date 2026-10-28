import fakeredis
import pytest
from fastapi.testclient import TestClient

from sentinel.api import create_app
from sentinel.models import Decision
from sentinel.queue import STREAM
from sentinel.store import Store

BODY = {
    "message_id": "m-1",
    "conversation_id": "c-1",
    "author_id": "u-1",
    "text": "hola",
    "created_at": "2026-10-28T12:00:00Z",
}


@pytest.fixture
def env(config):
    store, redis = Store(":memory:"), fakeredis.FakeRedis(decode_responses=True)
    client_a, _ = store.create_client("a", None)
    client_b, _ = store.create_client("b", None)
    keys = {
        "a": store.create_api_key(client_a, ["messages:write", "moderation:read", "review:write"])[1],
        "a_write_only": store.create_api_key(client_a, ["messages:write"])[1],
        "b": store.create_api_key(client_b, ["messages:write", "moderation:read"])[1],
    }
    with TestClient(create_app(config, store, redis)) as http:
        yield http, store, redis, keys


def auth(key):
    return {"Authorization": f"Bearer {key}"}


def test_post_message_queues_once(env):
    http, _, redis, keys = env
    first = http.post("/messages", json=BODY, headers=auth(keys["a"]))
    again = http.post("/messages", json=BODY, headers=auth(keys["a"]))
    assert (first.status_code, first.json()["status"]) == (202, "queued")
    assert (again.status_code, again.json()["status"]) == (202, "duplicate")
    assert redis.xlen(STREAM) == 1


def test_auth_errors(env):
    http, store, _, keys = env
    assert http.post("/messages", json=BODY).status_code == 401
    assert http.post("/messages", json=BODY, headers=auth("sk_wrong")).status_code == 401
    assert http.get("/review-queue", headers=auth(keys["a_write_only"])).status_code == 403
    key_id = store.list_api_keys()[0]["id"]
    store.revoke_api_key(key_id)
    assert http.post("/messages", json=BODY, headers=auth(keys["a"])).status_code == 401


def test_clients_are_isolated_and_review_flow(env):
    http, store, _, keys = env
    http.post("/messages", json=BODY, headers=auth(keys["a"]))
    assert http.get("/moderation/m-1", headers=auth(keys["a"])).json()["status"] == "pending"
    message_key = store._db.execute("SELECT id FROM messages").fetchone()["id"]
    store.save_verdict(message_key, Decision("flag", "insult", "high risk", "encoders", 0.9))

    assert http.get("/moderation/m-1", headers=auth(keys["b"])).status_code == 404
    assert http.get("/review-queue", headers=auth(keys["b"])).json()["items"] == []

    queue = http.get("/review-queue", headers=auth(keys["a"])).json()["items"]
    assert [i["message_id"] for i in queue] == ["m-1"]
    review = http.post("/review/m-1", json={"decision": "reject"}, headers=auth(keys["a"])).json()
    assert (review["model_verdict"], review["final_verdict"]) == ("flag", "ok")
    assert http.get("/review-queue", headers=auth(keys["a"])).json()["items"] == []
