"""Public API (modular monolith). Blueprint services map to: connector-service (import job), local-indexer + embedding-service
(reindex), entity-resolution-service (engine.people / engine.places), timeline-service, graph-service, search-service (hybrid
search + the multi-hop answer) and policy-service (purge). ocr-service is not built: receipts arrive as already-scanned text.

    uvicorn mem.api:app          python -m core.jobs mem.api      # the worker
"""
import datetime
import json
import pickle
import re
import uuid

from fastapi import Depends, Header, Query
from fastapi.encoders import jsonable_encoder
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import engine, world
from .seed import TENANTS

READ = {"archive:read", "jobs:read"}
PERMISSIONS = {"reader": READ, "owner": READ | {"archive:import", "archive:purge", "audit:read"}}
app = create_app("life-search-engine", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
UTC = datetime.UTC
_embedders = {}


# ---------------------------------------------------------------- connector-service + local-indexer

class ImportIn(BaseModel):
    source: str = Field("demo-archive", pattern="^demo-archive$", description="The one connector in this build: a deterministic synthetic export of photos, mail, calendar, receipts and messages")


@app.post("/v1/connectors/import", status_code=202, tags=["import"], summary="Import an archive export and index it (job)")
def start_import(body: ImportIn, ctx: Ctx = Depends(auth("archive:import")), idem: str | None = IdemKey):
    return run(ctx, idem, body, lambda c: (202, jobs.enqueue(c, ctx, "import.run", body.model_dump())))


def store(c, t, arts):
    """Insert the records not already present (same checksum = same record) and not purged earlier. -> how many were new"""
    have = {r["checksum"] for r in c.execute("SELECT checksum FROM artifacts UNION ALL SELECT checksum FROM tombstones")}
    new = [a for a in arts if a["checksum"] not in have]
    db.load(c, "artifacts", ["id", "tenant_id", "kind", "source", "uri", "captured_at", "checksum", "metadata", "ends_at", "body"],
            [(uuid.uuid5(t, a["checksum"]), t, a["kind"], a["source"], a["uri"], a["at"], a["checksum"], json.dumps(a["meta"]), a["ends_at"], a["text"]) for a in new])
    return len(new)


def reindex(c, t):
    """Rebuild everything derived from the records: embeddings, entities, mentions, moments and trips."""
    arts = c.execute("SELECT id, kind, captured_at AS at, ends_at, body AS text, metadata AS meta FROM artifacts ORDER BY captured_at, id").fetchall()
    for table in ("event_artifacts", "events", "mentions", "entities", "embeddings"):
        c.execute(f"DELETE FROM {table}")
    docs = [engine.doc(a) for a in arts]
    model = engine.embedder(docs)
    c.execute("""INSERT INTO indexes (tenant_id, embedder, fitted_on, version) VALUES (%s,%s,%s,%s)
                 ON CONFLICT (tenant_id) DO UPDATE SET embedder = excluded.embedder, fitted_on = excluded.fitted_on, version = excluded.version, fitted_at = now()""", [t, pickle.dumps(model), len(arts), engine.VERSION])
    db.load(c, "embeddings", ["artifact_id", "vector", "tenant_id"], [(a["id"], engine.literal(v), t) for a, v in zip(arts, model.transform(docs))])

    eid = lambda kind, key: uuid.uuid5(t, f"{kind}|{key}")      # noqa: E731
    groups = engine.moments(arts)
    found, of = engine.places(arts)
    links, unresolved = engine.people(arts, groups)
    ents, ment = {}, {}

    def mention(i, e, conf, how):
        if conf > ment.get((arts[i]["id"], e), (0,))[0]:
            ment[arts[i]["id"], e] = (conf, how)
    person = {i: eid("person", a["meta"]["email"]) for i, a in enumerate(arts) if a["kind"] == "contact"}
    for i, e in person.items():
        ents[e] = ("person", arts[i]["meta"]["name"], {"email": arts[i]["meta"]["email"], "nick": arts[i]["meta"].get("nick")})
    place = {key: eid("place", f"{key[0]}|{key[1]}") for key in found}
    for key, p in found.items():
        ents[place[key]] = ("place", p["name"], {"city": p["city"], "cuisine": p["cuisine"], "receipts": p["receipts"]})
    for i, c_idx, conf, how in links:
        mention(i, person[c_idx], conf, how)
    for i, key in of.items():
        mention(i, place[key], 1.0, "merchant line of the receipt")
    for i, a in enumerate(arts):
        booked = re.search(r"Your trip to (.+?) is confirmed", a["text"]) if a["kind"] == "email" else None
        city = booked[1] if booked else engine.city_of(a)
        if city in world.CITIES:
            ents[eid("city", city)] = ("city", city, {"lat": world.CITIES[city][0], "lon": world.CITIES[city][1]})
            mention(i, eid("city", city), 1.0 if a["kind"] != "photo" else 0.9, "booking e-mail" if booked else "address on the receipt" if a["kind"] == "receipt" else "photo GPS")

    events, ea, trips = [], {}, []
    for tr in engine.trips(arts):
        tid = eid("trip", f"{tr['city']}|{tr['start']}")
        start, end = datetime.datetime.combine(tr["start"], datetime.time.min, UTC), datetime.datetime.combine(tr["end"], datetime.time.max, UTC)
        trips.append((tid, start, end))
        events.append((tid, t, "trip", start, end, eid("city", tr["city"]), f"{tr['city']}, {(tr['end'] - tr['start']).days + 1} days", None))
        for a in arts:
            if a["kind"] in engine.MOMENT_KINDS and start <= a["at"] <= end:
                ea[tid, a["id"]] = 1.0
            elif a["kind"] == "email" and f"trip to {tr['city']}" in a["text"] and (m := re.search(r"Departing (\d{4}-\d\d-\d\d)", a["text"])) and abs((datetime.date.fromisoformat(m[1]) - tr["start"]).days) <= 1:
                ea[tid, a["id"]] = 0.8
    for g in groups:
        start, end = arts[g[0]]["at"], max(arts[i]["ends_at"] or arts[i]["at"] for i in g)
        receipts = [i for i in g if i in of]
        city = next((c_ for c_ in (engine.city_of(arts[i]) for i in g) if c_), None)
        tags = [tag for i in g for tag in arts[i]["meta"].get("tags", [])]
        top = sorted(set(tags), key=lambda x: (-tags.count(x), x))[:3]
        mid = eid("moment", arts[g[0]]["id"])
        events.append((mid, t, "moment", start, end, place[of[receipts[0]]] if receipts else eid("city", city) if city else None,
                       found[of[receipts[0]]]["name"] if receipts else ", ".join(top) or arts[g[0]]["text"][:60], next((tid for tid, a, b in trips if a <= start <= b), None)))
        for i in g:
            ea[mid, arts[i]["id"]] = 1.0
            if receipts and arts[i]["kind"] in ("calendar", "message") and engine.fold(found[of[receipts[0]]]["name"]) in engine.fold(arts[i]["text"]):
                mention(i, place[of[receipts[0]]], 0.95, "names the place of this moment")
    db.many(c, "INSERT INTO entities (id, tenant_id, entity_type, canonical_name, attributes) VALUES (%s,%s,%s,%s,%s)", [(e, t, kind, name, Jsonb(attrs)) for e, (kind, name, attrs) in ents.items()])
    db.load(c, "mentions", ["artifact_id", "entity_id", "span", "confidence", "tenant_id"], [(a, e, json.dumps({"how": how}), conf, t) for (a, e), (conf, how) in ment.items()])
    db.load(c, "events", ["id", "tenant_id", "event_type", "starts_at", "ends_at", "place_id", "summary", "parent_id"], sorted(events, key=lambda e: e[2] != "trip"))     # trips first: moments point at them
    db.load(c, "event_artifacts", ["event_id", "artifact_id", "relevance", "tenant_id"], [(e, a, rel, t) for (e, a), rel in ea.items()])
    kinds = {}
    for a in arts:
        kinds[a["kind"]] = kinds.get(a["kind"], 0) + 1
    return {"records": len(arts), "by_kind": kinds, "people": len(person), "places": len(found), "cities": sum(v[0] == "city" for v in ents.values()), "trips": len(trips), "moments": len(groups),
            "mentions": len(ment), "first_name_mentions_left_unresolved": unresolved, "embedding": f"local LSA, {engine.DIM} dimensions, fitted on this archive only", "index_version": engine.VERSION}


@jobs.handler("import.run")
def import_job(c, job):
    t = job["tenant_id"]
    name = c.execute("SELECT name FROM tenants WHERE id = %s", [t]).fetchone()["name"]
    seed, n_trips = next((s, k) for n, s, k in TENANTS.values() if n == name)
    w = world.build(seed, n_trips=n_trips)
    new = store(c, t, w["artifacts"])
    if not new:
        return {"new_records": 0, "note": "every record in this export is already in the archive or was purged by the owner; nothing changed"}
    out = {"new_records": new, "sources": sorted({a["source"] for a in w["artifacts"]}), "from": w["artifacts"][0]["at"], "to": w["artifacts"][-1]["at"], **reindex(c, t)}
    audit.record(c, Ctx(t, uuid.UUID(job["payload"]["actor_id"]), "worker", "system"), "import.completed", "tenant", t, {"new_records": new, "index_version": engine.VERSION})
    return out


# ---------------------------------------------------------------- search-service

def embed(c, t, text):
    row = c.execute("SELECT fitted_at FROM indexes").fetchone()
    if not row:
        raise Problem(409, "nothing_imported", "import an archive first: POST /v1/connectors/import")
    if _embedders.get(t, (None,))[0] != row["fitted_at"]:
        _embedders[t] = (row["fitted_at"], pickle.loads(c.execute("SELECT embedder FROM indexes").fetchone()["embedder"]))      # fitted and stored by this service
    return engine.literal(_embedders[t][1].transform([text])[0])


def nearest(c, qvec, k, where="", params=()):
    """Vector search with filters. Iterative scan keeps the HNSW index from returning too few rows when the filter is narrow."""
    c.execute("SET LOCAL hnsw.iterative_scan = strict_order")
    return c.execute(f"""SELECT a.id, a.kind, a.source, a.captured_at, a.body, a.metadata, round((1 - (e.vector <=> %s::vector))::numeric, 4)::float AS similarity
                           FROM embeddings e JOIN artifacts a ON a.id = e.artifact_id WHERE true {where} ORDER BY e.vector <=> %s::vector LIMIT %s""", [qvec, *params, qvec, k]).fetchall()


def brief(a):
    return {"id": a["id"], "kind": a["kind"], "captured_at": a["captured_at"], "text": a["body"][:160], **({"similarity": a["similarity"]} if "similarity" in a else {})}


def ask(c, t, question):
    """The multi-hop answer: trip -> photo -> moment -> restaurant -> receipt and people. Every hop is recorded with the
    records it rests on, and a hop that finds nothing ends the chain with the reason instead of a guess."""
    p = engine.parse(question)
    if not (p["object"] or p["cuisine"]):
        return None
    trace, when, params, trip = [], "", [], None
    stop = lambda reason: {"status": "not_found", "reason": reason, "parsed": p, "trace": trace}      # noqa: E731
    if p["city"]:
        found = c.execute("""SELECT e.id, e.starts_at, e.ends_at, e.summary FROM events e JOIN entities n ON n.id = e.place_id
                              WHERE e.event_type = 'trip' AND n.canonical_name = %s AND (%s::int IS NULL OR extract(year FROM e.starts_at) = %s) ORDER BY e.starts_at""", [p["city"], p["year"], p["year"]]).fetchall()
        trace.append({"step": "Find the trip", "looked_for": f"a trip to {p['city']}" + (f" in {p['year']}" if p["year"] else ""), "found": [f"{x['summary']}, from {x['starts_at']:%d %b %Y}" for x in found], "sources": [x["id"] for x in found]})
        if not found:
            return stop(f"no trip to {p['city']}" + (f" in {p['year']}" if p["year"] else "") + " is in this archive")
        trip = found[0] if len(found) == 1 else None
        when = " AND (" + " OR ".join("a.captured_at BETWEEN %s AND %s" for _ in found) + ")"
        params = [v for x in found for v in (x["starts_at"], x["ends_at"])]
    elif p["year"]:
        when, params = " AND a.captured_at >= %s AND a.captured_at < %s", [datetime.datetime(p["year"], 1, 1, tzinfo=UTC), datetime.datetime(p["year"] + 1, 1, 1, tzinfo=UTC)]
    moment_sql = """SELECT e.id, e.starts_at, e.ends_at, e.parent_id, n.id AS place_id, n.canonical_name AS place, n.entity_type, n.attributes FROM event_artifacts ea
                      JOIN events e ON e.id = ea.event_id AND e.event_type = 'moment' LEFT JOIN entities n ON n.id = e.place_id WHERE ea.artifact_id = %s"""
    fits = lambda m: m and m["entity_type"] == "place" and (not p["cuisine"] or m["attributes"].get("cuisine") == p["cuisine"])      # noqa: E731
    photo = moment = None
    if p["object"]:
        photos = [x for x in nearest(c, embed(c, t, "photo " + p["object"]), 6, " AND a.kind = 'photo'" + when, params) if x["similarity"] >= engine.MIN_SIMILARITY]
        trace.append({"step": "Find the photo", "looked_for": f"a photo of a {p['object']}" + (" during that trip" if p["city"] else ""), "found": [f"{' · '.join(x['metadata']['tags'])} ({x['similarity']:.2f})" for x in photos[:3]], "sources": [x["id"] for x in photos[:3]]})
        if not photos:
            return stop(f"no photo resembling a {p['object']} was found" + (" during that trip" if p["city"] else ""))
        for x in photos:
            m = c.execute(moment_sql, [x["id"]]).fetchone()
            if fits(m):
                photo, moment = x, m
                break
        trace.append({"step": "Find where it was taken", "looked_for": f"the moment that photo belongs to, at {'an' if (p['cuisine'] or 'x')[0] in 'AEIOU' else 'a'} {p['cuisine'] or ''} restaurant".replace("  ", " "),
                      "found": [f"{moment['place']} ({moment['attributes'].get('cuisine')}), {moment['starts_at']:%d %b %Y %H:%M} to {moment['ends_at']:%H:%M}"] if moment else [], "sources": [moment["id"]] if moment else []})
        if not moment:
            return stop(f"a matching photo exists, but it was not taken at a {p['cuisine'] or ''} restaurant visit".replace("  ", " "))
    else:
        rows = c.execute(f"""SELECT DISTINCT e.id, e.starts_at, e.ends_at, e.parent_id, n.id AS place_id, n.canonical_name AS place, n.entity_type, n.attributes FROM events e JOIN entities n ON n.id = e.place_id
                              JOIN event_artifacts ea ON ea.event_id = e.id JOIN artifacts a ON a.id = ea.artifact_id
                             WHERE e.event_type = 'moment' AND n.entity_type = 'place' AND n.attributes->>'cuisine' = %s {when} ORDER BY e.starts_at""", [p["cuisine"], *params]).fetchall()
        trace.append({"step": "Find the meal", "looked_for": f"{p['cuisine']} restaurant visits in that period", "found": [f"{m['place']}, {m['starts_at']:%d %b %Y}" for m in rows[:5]], "sources": [m["id"] for m in rows[:5]]})
        if len(rows) != 1:
            return stop(f"{len(rows)} {p['cuisine']} restaurant visits fit; say more (a photo, a person, a date) to pick one" if rows else f"no {p['cuisine']} restaurant visit was found in that period")
        moment = rows[0]
    inside = c.execute("SELECT a.id, a.kind, a.source, a.captured_at, a.body, a.metadata FROM event_artifacts ea JOIN artifacts a ON a.id = ea.artifact_id WHERE ea.event_id = %s ORDER BY a.captured_at", [moment["id"]]).fetchall()
    receipts = [a for a in inside if a["kind"] == "receipt"]
    who = c.execute("""SELECT n.id AS entity_id, n.canonical_name AS name, max(m.confidence) AS confidence, (array_agg(m.span->>'how' ORDER BY m.confidence DESC))[1] AS how, array_agg(m.artifact_id) AS sources
                         FROM event_artifacts ea JOIN mentions m ON m.artifact_id = ea.artifact_id JOIN entities n ON n.id = m.entity_id AND n.entity_type = 'person'
                        WHERE ea.event_id = %s GROUP BY n.id ORDER BY confidence DESC, name""", [moment["id"]]).fetchall()
    trace.append({"step": "Find the receipt", "looked_for": "a receipt in the same moment", "found": [f"{r['body'].splitlines()[0]}, total ${engine.receipt_total(r['body']):.2f}" for r in receipts], "sources": [r["id"] for r in receipts]})
    trace.append({"step": "Find who was there", "looked_for": "people named in that moment's calendar entry and messages", "found": [f"{x['name']} ({x['how']})" for x in who], "sources": sorted({s for x in who for s in x["sources"]}, key=str)})
    trip = trip or (c.execute("SELECT id, starts_at, ends_at, summary FROM events WHERE id = %s", [moment["parent_id"]]).fetchone() if moment["parent_id"] else None)
    r = receipts[0] if receipts else None
    sentence = f"{moment['place']}" + (f" ({moment['attributes']['cuisine']})" if moment["attributes"].get("cuisine") else "") + f", {moment['starts_at']:%A %d %B %Y}."
    sentence += f" The receipt total was ${engine.receipt_total(r['body']):.2f}." if r else " No receipt for that meal is in the archive."
    sentence += (" You were with " + " and ".join(x["name"] for x in who) + ".") if who else " The archive does not say who you were with."
    return {"status": "answered", "sentence": sentence, "parsed": p, "trip": trip, "photo": {**brief(photo), "tags": photo["metadata"]["tags"], "gps": photo["metadata"]["gps"]} if photo else None,
            "restaurant": {"entity_id": moment["place_id"], "name": moment["place"], "cuisine": moment["attributes"].get("cuisine"), "city": moment["attributes"].get("city")},
            "moment": {"id": moment["id"], "starts_at": moment["starts_at"], "ends_at": moment["ends_at"], "records": [brief(a) for a in inside]},
            "receipt": {"id": r["id"], "total": engine.receipt_total(r["body"]), "captured_at": r["captured_at"], "text": r["body"]} if r else None, "people": who, "trace": trace,
            "sources": sorted({a["id"] for a in inside}, key=str)}


class SearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=400)
    k: int = Field(8, ge=1, le=50)
    kinds: list[str] | None = Field(None, max_length=6)


def hybrid(c, t, query, k, kinds=None):
    """Vector similarity and full-text rank, fused by reciprocal rank."""
    words = re.findall(r"[a-z0-9]+", query.lower())
    rows = c.execute("""
        WITH v AS (SELECT e.artifact_id AS id, row_number() OVER (ORDER BY e.vector <=> %(q)s::vector) AS r FROM embeddings e JOIN artifacts a ON a.id = e.artifact_id
                    WHERE %(kinds)s::text[] IS NULL OR a.kind = ANY(%(kinds)s) ORDER BY e.vector <=> %(q)s::vector LIMIT 40),
             f AS (SELECT a.id, row_number() OVER (ORDER BY ts_rank(a.tsv, to_tsquery('english', %(ts)s)) DESC) AS r FROM artifacts a
                    WHERE a.tsv @@ to_tsquery('english', %(ts)s) AND (%(kinds)s::text[] IS NULL OR a.kind = ANY(%(kinds)s)) ORDER BY 2 LIMIT 40)
        SELECT a.id, a.kind, a.source, a.captured_at, a.body, round(sum(1.0 / (60 + x.r)), 5)::float AS score, bool_or(x.src = 'v') AS by_meaning, bool_or(x.src = 'f') AS by_words
          FROM (SELECT id, r, 'v' AS src FROM v UNION ALL SELECT id, r, 'f' FROM f) x JOIN artifacts a ON a.id = x.id GROUP BY a.id ORDER BY score DESC, a.id LIMIT %(k)s""",
                     {"q": embed(c, t, query), "ts": " | ".join(words) or "x", "kinds": kinds, "k": k}).fetchall()
    return [{**brief(r), "score": r["score"], "matched_by": [n for n, on in (("meaning", r["by_meaning"]), ("words", r["by_words"])) if on]} for r in rows]


@app.post("/v1/search", tags=["search"], summary="Search the archive. A question that names a photo or a cuisine is also answered step by step, with the records each step rests on")
def search(body: SearchIn, ctx: Ctx = Depends(auth("archive:read"))):
    with db.tx(ctx.tenant_id) as c:
        answer = ask(c, ctx.tenant_id, body.query)
        return jsonable_encoder({"query": body.query, "answer": answer, "results": hybrid(c, ctx.tenant_id, body.query, body.k, body.kinds),
                                 "plain_vector_search": [brief(a) for a in nearest(c, embed(c, ctx.tenant_id, body.query), 5)]})


# ---------------------------------------------------------------- timeline-service + graph-service

@app.get("/v1/timeline", tags=["timeline"], summary="Trips or moments in a period, oldest first, each with its place, people and record counts")
def timeline(level: str = "trip", limit: int = 20, after: uuid.UUID | None = None, ctx: Ctx = Depends(auth("archive:read")),
             frm: datetime.datetime | None = Query(None, alias="from"), to: datetime.datetime | None = None):
    if level not in ("trip", "moment"):
        raise Problem(422, "bad_level", "level is 'trip' or 'moment'")
    limit = max(1, min(limit, 100))
    with db.tx(ctx.tenant_id) as c:
        rows = c.execute("""SELECT e.id, e.event_type, e.starts_at, e.ends_at, e.summary, n.canonical_name AS place, n.entity_type AS place_type,
                                   (SELECT count(*) FROM event_artifacts ea WHERE ea.event_id = e.id) AS records,
                                   (SELECT count(*) FROM events k WHERE k.parent_id = e.id) AS moments,
                                   (SELECT coalesce(array_agg(DISTINCT p.canonical_name), '{}') FROM event_artifacts ea JOIN mentions m ON m.artifact_id = ea.artifact_id JOIN entities p ON p.id = m.entity_id AND p.entity_type = 'person' WHERE ea.event_id = e.id) AS people
                              FROM events e LEFT JOIN entities n ON n.id = e.place_id
                             WHERE e.event_type = %s AND (%s::timestamptz IS NULL OR e.ends_at >= %s) AND (%s::timestamptz IS NULL OR e.starts_at <= %s)
                               AND (%s::uuid IS NULL OR (e.starts_at, e.id) > (SELECT starts_at, id FROM events WHERE id = %s))
                             ORDER BY e.starts_at, e.id LIMIT %s""", [level, frm, frm, to, to, after, after, limit + 1]).fetchall()
    return jsonable_encoder({"items": rows[:limit], "next_after": rows[limit - 1]["id"] if len(rows) > limit else None})


@app.get("/v1/entities", tags=["graph"], summary="(+) Find people, places and cities by name")
def entities(q: str = "", type: str | None = None, ctx: Ctx = Depends(auth("archive:read"))):
    with db.tx(ctx.tenant_id) as c:
        return jsonable_encoder({"items": c.execute("""SELECT n.id, n.entity_type, n.canonical_name, n.attributes, (SELECT count(*) FROM mentions m WHERE m.entity_id = n.id) AS mentions FROM entities n
                                                        WHERE n.canonical_name ILIKE %s AND (%s::text IS NULL OR n.entity_type = %s) ORDER BY mentions DESC, n.canonical_name LIMIT 30""", [f"%{q}%", type, type]).fetchall()})


@app.get("/v1/entities/{entity_id}/graph", tags=["graph"], summary="A person, place or city with what it is connected to: shared moments, first and last seen")
def graph(entity_id: uuid.UUID, ctx: Ctx = Depends(auth("archive:read"))):
    with db.tx(ctx.tenant_id) as c:
        n = c.execute("SELECT id, entity_type, canonical_name, attributes FROM entities WHERE id = %s", [entity_id]).fetchone()
        if not n:
            raise Problem(404, "entity_not_found")
        mine = """SELECT DISTINCT ea.event_id FROM mentions m JOIN event_artifacts ea ON ea.artifact_id = m.artifact_id JOIN events e ON e.id = ea.event_id AND e.event_type = 'moment' WHERE m.entity_id = %s"""
        seen = c.execute(f"SELECT count(*) AS moments, min(e.starts_at) AS first_seen, max(e.starts_at) AS last_seen FROM events e WHERE e.id IN ({mine})", [entity_id]).fetchone()
        edges = c.execute(f"""SELECT o.id, o.entity_type, o.canonical_name, count(DISTINCT ea.event_id) AS shared_moments, max(e.starts_at) AS last_together
                                FROM event_artifacts ea JOIN events e ON e.id = ea.event_id JOIN mentions m ON m.artifact_id = ea.artifact_id JOIN entities o ON o.id = m.entity_id
                               WHERE ea.event_id IN ({mine}) AND o.id <> %s GROUP BY o.id ORDER BY shared_moments DESC, o.canonical_name LIMIT 12""", [entity_id, entity_id]).fetchall()
        how = c.execute("SELECT m.span->>'how' AS how, count(*) AS n, round(avg(m.confidence)::numeric, 2)::float AS confidence FROM mentions m WHERE m.entity_id = %s GROUP BY 1 ORDER BY n DESC", [entity_id]).fetchall()
        recent = c.execute(f"SELECT e.id, e.starts_at, e.summary FROM events e WHERE e.id IN ({mine}) ORDER BY e.starts_at DESC LIMIT 5", [entity_id]).fetchall()
    return jsonable_encoder({"entity": n, **seen, "linked_by": how, "connected_to": edges, "recent_moments": recent})


@app.get("/v1/artifacts/{artifact_id}", tags=["archive"], summary="(+) One source record, with the entities found in it and the events it belongs to")
def artifact(artifact_id: uuid.UUID, ctx: Ctx = Depends(auth("archive:read"))):
    with db.tx(ctx.tenant_id) as c:
        a = c.execute("SELECT id, kind, source, uri, captured_at, ends_at, checksum, metadata, body FROM artifacts WHERE id = %s", [artifact_id]).fetchone()
        if not a:
            raise Problem(404, "artifact_not_found")
        ents = c.execute("SELECT n.id, n.entity_type, n.canonical_name, m.confidence, m.span->>'how' AS how FROM mentions m JOIN entities n ON n.id = m.entity_id WHERE m.artifact_id = %s ORDER BY m.confidence DESC", [artifact_id]).fetchall()
        evs = c.execute("SELECT e.id, e.event_type, e.starts_at, e.summary FROM event_artifacts ea JOIN events e ON e.id = ea.event_id WHERE ea.artifact_id = %s ORDER BY e.event_type", [artifact_id]).fetchall()
    return jsonable_encoder({**a, "entities": ents, "events": evs})


# ---------------------------------------------------------------- summaries

class SummarizeIn(BaseModel):
    event_id: uuid.UUID


@app.post("/v1/memories/summarize", tags=["search"], summary="Summarise a trip or a moment from its records. Extractive: every sentence lists the records it was built from")
def summarize(body: SummarizeIn, ctx: Ctx = Depends(auth("archive:read"))):
    with db.tx(ctx.tenant_id) as c:
        e = c.execute("SELECT e.id, e.event_type, e.starts_at, e.ends_at, e.summary, n.canonical_name AS place FROM events e LEFT JOIN entities n ON n.id = e.place_id WHERE e.id = %s", [body.event_id]).fetchone()
        if not e:
            raise Problem(404, "event_not_found")
        arts = c.execute("SELECT a.id, a.kind, a.captured_at, a.body, a.metadata, ea.relevance FROM event_artifacts ea JOIN artifacts a ON a.id = ea.artifact_id WHERE ea.event_id = %s ORDER BY a.captured_at", [e["id"]]).fetchall()
        who = c.execute("""SELECT n.canonical_name AS name, count(DISTINCT m.artifact_id) AS n, array_agg(DISTINCT m.artifact_id) AS sources FROM event_artifacts ea JOIN mentions m ON m.artifact_id = ea.artifact_id
                             JOIN entities n ON n.id = m.entity_id AND n.entity_type = 'person' WHERE ea.event_id = %s GROUP BY n.id ORDER BY n DESC, name""", [e["id"]]).fetchall()
    out = []
    say = lambda text, sources: out.append({"text": text, "sources": sorted(set(sources), key=str)})      # noqa: E731
    booking = [a for a in arts if a["kind"] == "email"]
    days = (e["ends_at"].date() - e["starts_at"].date()).days + 1
    say(f"{e['place'] or e['summary']}, {e['starts_at']:%d %B %Y}" + (f" to {e['ends_at']:%d %B %Y} ({days} days)." if days > 1 else f", {e['starts_at']:%H:%M} to {e['ends_at']:%H:%M}."), [a["id"] for a in booking] or [arts[0]["id"], arts[-1]["id"]] if arts else [])
    if who:
        say("With " + ", ".join(x["name"] for x in who) + ".", [s for x in who for s in x["sources"]])
    receipts = [a for a in arts if a["kind"] == "receipt" and engine.receipt_total(a["body"]) is not None]
    if receipts:
        top = sorted(receipts, key=lambda a: -engine.receipt_total(a["body"]))[:3]
        say(f"{len(receipts)} meals on receipts, ${sum(engine.receipt_total(a['body']) for a in receipts):,.2f} in total. The largest: " + "; ".join(f"{engine.fold(a['body'].splitlines()[0]).title()} ${engine.receipt_total(a['body']):.2f}" for a in top) + ".", [a["id"] for a in receipts])
    photos = [a for a in arts if a["kind"] == "photo"]
    if photos:
        tags = [t for a in photos for t in a["metadata"].get("tags", [])]
        say(f"{len(photos)} photos. Most often in them: " + ", ".join(sorted(set(tags), key=lambda t: (-tags.count(t), t))[:5]) + ".", [a["id"] for a in photos])
    kept = [s for s in out if s["sources"]]                      # a sentence with nothing behind it is dropped, not shown
    return jsonable_encoder({"event": e, "method": "extractive, rule-based; no language model", "sentences": kept, "dropped_for_lack_of_sources": len(out) - len(kept), "records": len(arts)})


# ---------------------------------------------------------------- policy-service

@app.delete("/v1/artifacts/{artifact_id}", tags=["archive"], summary="Purge one record and everything derived from it (purge=true is required: there is no soft delete)")
def purge(artifact_id: uuid.UUID, purge: bool = False, ctx: Ctx = Depends(auth("archive:purge")), idem: str | None = IdemKey):
    if not purge:
        raise Problem(422, "purge_required", "this endpoint only purges: pass purge=true")

    def work(c):
        a = c.execute("DELETE FROM artifacts WHERE id = %s RETURNING kind, checksum", [artifact_id]).fetchone()          # mentions, embedding and event links go with it (ON DELETE CASCADE)
        if not a:
            raise Problem(404, "artifact_not_found")
        c.execute("INSERT INTO tombstones (tenant_id, checksum) VALUES (%s,%s)", [ctx.tenant_id, a["checksum"]])
        emptied = c.execute("DELETE FROM events e WHERE NOT EXISTS (SELECT 1 FROM event_artifacts ea WHERE ea.event_id = e.id) RETURNING id").fetchall()
        orphaned = c.execute("DELETE FROM entities n WHERE n.entity_type = 'place' AND NOT EXISTS (SELECT 1 FROM mentions m WHERE m.entity_id = n.id) RETURNING id").fetchall()
        audit.record(c, ctx, "artifact.purged", "artifact", artifact_id, {"kind": a["kind"], "checksum_prefix": a["checksum"][:12]})      # the audit trail keeps no content
        left = c.execute("SELECT (SELECT count(*) FROM embeddings WHERE artifact_id = %s) + (SELECT count(*) FROM mentions WHERE artifact_id = %s) + (SELECT count(*) FROM event_artifacts WHERE artifact_id = %s) AS n",
                         [artifact_id] * 3).fetchone()["n"]
        return 200, {"purged": artifact_id, "kind": a["kind"], "derived_rows_left": left, "events_removed": len(emptied), "places_removed": len(orphaned)}
    return run(ctx, idem, {"id": str(artifact_id)}, work)
