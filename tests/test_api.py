"""Integration, end-to-end and security tests against a real Postgres with pgvector."""
import numpy as np
import pytest

from core import db, jobs, scenario
from mem import api, world

from .conftest import bearer

OWNER, READER, OTHER = bearer("maya", "owner"), bearer("maya", "reader"), bearer("other", "owner")
Q = "Find the Italian restaurant from my 2024 Chicago trip where I photographed a red motorcycle, then show the receipt and who I was with."


def test_search_before_any_import_says_so(client):
    assert client.post("/v1/search", headers=READER, json={"query": Q}).json()["code"] == "nothing_imported"


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


@pytest.fixture(scope="module")
def truth():
    w = world.build(11)
    return w, next(q for q in world.questions(w) if (q["city"], q["year"], q["object"]) == ("Chicago", 2024, "red motorcycle"))


def test_demo_journey_matches_the_generated_archive(demo, truth):
    w, q = truth
    imp = demo["imp"]["result"]
    assert imp["new_records"] == imp["records"] == len(w["artifacts"]) and imp["trips"] == len(w["trips"]) == len(demo["trips"]["items"]) and imp["people"] == len(w["people"])
    assert {(t["place"], t["starts_at"][:10]) for t in demo["trips"]["items"]} == {(t["city"], t["start"].isoformat()) for t in w["trips"]}
    a = demo["ask"]["answer"]
    receipt = next(x for x in w["artifacts"] if x["checksum"] == q["receipt"])
    assert a["status"] == "answered" and a["restaurant"]["name"] == q["restaurant"] == "Osteria Bellavita" and a["restaurant"]["cuisine"] == "Italian"
    assert a["receipt"]["total"] == receipt["meta"]["total"] and a["receipt"]["text"] == receipt["text"] and {p["name"] for p in a["people"]} == q["people"]
    assert a["photo"]["tags"][0] == "red" and [t["step"] for t in a["trace"]] == ["Find the trip", "Find the photo", "Find where it was taken", "Find the receipt", "Find who was there"]
    assert all(t["sources"] for t in a["trace"]) and a["receipt"]["id"] in a["sources"] and a["photo"]["id"] in a["sources"]
    assert demo["ask"]["plain_vector_search"][0]["id"] != a["receipt"]["id"]                       # the whole sentence, embedded, does not find it
    assert demo["receipt"]["body"] == receipt["text"] and "Osteria Bellavita" in {e["canonical_name"] for e in demo["receipt"]["entities"]}
    assert demo["person"]["entity"]["canonical_name"] in q["people"] and demo["person"]["moments"] >= 1
    s = demo["sum"]
    assert s["dropped_for_lack_of_sources"] == 0 and len(s["sentences"]) == 4 and all(x["sources"] for x in s["sentences"]) and "Chicago" in s["sentences"][0]["text"]
    assert demo["purge"]["derived_rows_left"] == 0
    again = demo["ask2"]["answer"]
    assert again["status"] == "answered" and again["receipt"] is None and "No receipt" in again["sentence"] and again["people"] == a["people"]
    assert demo["ask3"]["answer"]["status"] == "not_found" and "no trip to Boston in 2019" in demo["ask3"]["answer"]["reason"]
    purged = next(e for e in demo["audit"]["events"] if e["action"] == "artifact.purged")
    assert demo["audit"]["chain_valid"] and set(purged["detail"]) == {"kind", "checksum_prefix"}                 # no content in the audit trail


def test_the_purged_text_is_gone_from_every_table(demo, truth):
    with db.tx() as c:
        assert c.execute("SELECT count(*) AS n FROM artifacts WHERE checksum = %s", [truth[1]["receipt"]]).fetchone()["n"] == 0
        assert c.execute("SELECT count(*) AS n FROM audit_events WHERE detail::text ILIKE '%%RISOTTO%%' OR detail::text ILIKE '%%153.6%%'").fetchone()["n"] == 0
        assert c.execute("SELECT count(*) AS n FROM embeddings e LEFT JOIN artifacts a ON a.id = e.artifact_id WHERE a.id IS NULL").fetchone()["n"] == 0


def test_importing_the_same_export_again_adds_nothing_and_does_not_bring_back_a_purged_record(client, demo):
    with db.tx() as c:
        before = c.execute("SELECT (SELECT count(*) FROM artifacts) a, (SELECT count(*) FROM events) e, (SELECT count(*) FROM mentions) m").fetchone()
    job = client.post("/v1/connectors/import", headers=OWNER, json={}).json()
    jobs.drain()
    out = client.get(job["status_url"], headers=OWNER).json()["result"]
    with db.tx() as c:
        after = c.execute("SELECT (SELECT count(*) FROM artifacts) a, (SELECT count(*) FROM events) e, (SELECT count(*) FROM mentions) m").fetchone()
    assert out["new_records"] == 0 and after == before                              # the receipt purged in the demo stays purged


def test_filtered_vector_search_matches_brute_force(demo):
    """A narrow date filter over an HNSW index can silently return too few rows; the iterative scan must not."""
    t = demo["trips"]["items"][3]
    with db.tx() as c:
        tenant = c.execute("SELECT id FROM tenants WHERE name LIKE 'Maya%'").fetchone()["id"]
    with db.tx(tenant) as c:
        q = api.embed(c, tenant, "photo yellow tram")
        got = [r["id"] for r in api.nearest(c, q, 5, " AND a.kind = 'photo' AND a.captured_at BETWEEN %s AND %s", [t["starts_at"], t["ends_at"]])]
        rows = c.execute("SELECT a.id, e.vector::text AS v FROM embeddings e JOIN artifacts a ON a.id = e.artifact_id WHERE a.kind = 'photo' AND a.captured_at BETWEEN %s AND %s", [t["starts_at"], t["ends_at"]]).fetchall()
    qv = np.array(eval(q))
    sims = sorted(((float(np.array(eval(r["v"])) @ qv), r["id"]) for r in rows), reverse=True)[:5]
    assert len(got) == 5 and [round(s, 3) for s, _ in sims] == [round(float(np.array(eval(next(r["v"] for r in rows if r["id"] == g))) @ qv), 3) for g in got]


def test_timeline_pages_without_repeats_and_filters_by_date(client, demo):
    seen, after = [], None
    while True:
        page = client.get("/v1/timeline", headers=READER, params={"level": "moment", "limit": 100, **({"after": after} if after else {})}).json()
        seen += [e["id"] for e in page["items"]]
        after = page["next_after"]
        if not after:
            break
    assert len(seen) == len(set(seen)) == demo["imp"]["result"]["moments"]
    year = client.get("/v1/timeline", headers=READER, params={"from": "2023-01-01T00:00:00Z", "to": "2023-12-31T23:59:59Z"}).json()["items"]
    assert year and all(e["starts_at"].startswith("2023") for e in year)


def test_a_question_naming_only_a_cuisine_asks_for_more_when_several_fit(client, demo):
    a = client.post("/v1/search", headers=READER, json={"query": "Which Italian restaurant did I go to?"}).json()["answer"]
    assert a["status"] == "not_found" and "say more" in a["reason"]
    plain = client.post("/v1/search", headers=READER, json={"query": "itinerary confirmation", "kinds": ["email"]}).json()
    assert plain["answer"] is None and plain["results"] and all(r["kind"] == "email" for r in plain["results"])


@pytest.mark.parametrize("headers,method,path,body,expected", [
    ({}, "POST", "/v1/search", {"query": "x y"}, 401),
    (READER, "POST", "/v1/search", {"query": "red motorcycle"}, 200),
    (READER, "POST", "/v1/connectors/import", {}, 403),
    (READER, "DELETE", "/v1/artifacts/00000000-0000-0000-0000-000000000000?purge=true", None, 403),
    (READER, "GET", "/v1/audit", None, 403),
    (OWNER, "DELETE", "/v1/artifacts/00000000-0000-0000-0000-000000000000", None, 422),
    (OWNER, "DELETE", "/v1/artifacts/00000000-0000-0000-0000-000000000000?purge=true", None, 404),
    (OWNER, "POST", "/v1/connectors/import", {"source": "http://internal/"}, 422),
    (READER, "GET", "/v1/timeline?level=decade", None, 422),
    (READER, "POST", "/v1/search", {"query": "x' OR 1=1 --; DROP TABLE artifacts"}, 200),
    (READER, "POST", "/v1/memories/summarize", {"event_id": "00000000-0000-0000-0000-000000000000"}, 404),
])
def test_authorization_and_validation(client, demo, headers, method, path, body, expected):
    assert client.request(method, path, headers=headers, json=body).status_code == expected


def test_one_archive_cannot_reach_another(client, demo):
    mine = demo["ask"]["answer"]
    assert client.post("/v1/search", headers=OTHER, json={"query": Q}).json()["code"] == "nothing_imported"
    assert client.get(f"/v1/artifacts/{mine['photo']['id']}", headers=OTHER).status_code == 404
    assert client.get(f"/v1/entities/{mine['restaurant']['entity_id']}/graph", headers=OTHER).status_code == 404
    assert client.delete(f"/v1/artifacts/{mine['photo']['id']}?purge=true", headers=OTHER).status_code == 404
    job = client.post("/v1/connectors/import", headers=OTHER, json={}).json()
    jobs.drain()
    assert 0 < client.get(job["status_url"], headers=OTHER).json()["result"]["records"] < demo["imp"]["result"]["records"]
    theirs = client.post("/v1/search", headers=OTHER, json={"query": Q}).json()
    ids = {r["id"] for r in theirs["results"]} | {r["id"] for r in theirs["plain_vector_search"]} | set((theirs["answer"] or {}).get("sources", []))
    assert ids and not ids & (set(mine["sources"]) | {r["id"] for r in demo["ask"]["results"]})
    assert client.get(f"/v1/artifacts/{mine['photo']['id']}", headers=READER).status_code == 200        # and theirs did not disturb this one
