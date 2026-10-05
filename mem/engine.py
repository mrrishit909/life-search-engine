"""Indexing and query understanding. Pure functions over artifact dicts {kind, at, ends_at, text, meta}; no database,
no network: everything here would run on the owner's own device."""
import collections
import datetime
import math
import re

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from . import world

VERSION = "mem-1.0"
DIM = 128
GAP = datetime.timedelta(minutes=120)      # silence longer than this starts a new moment
SPLIT_KM = 0.6                             # so does moving this far between two located records
MOMENT_KINDS = {"photo", "receipt", "calendar", "message"}
MIN_SIMILARITY = 0.3                       # below this a photo is not offered as a match for the described object
UNFOLD = str.maketrans("1058", "IOSB")


# ---------------------------------------------------------------- embeddings

def doc(a):
    """The text that stands for a record in search."""
    m = a["meta"]
    if a["kind"] == "photo":
        return "photo " + " ".join(m.get("tags", []))
    return " ".join(str(x) for x in (m.get("subject"), m.get("from"), a["text"], m.get("location")) if x)


def embedder(texts):
    """Latent semantic analysis fitted on the owner's own archive: character n-gram TF-IDF reduced to DIM dimensions.
    Character n-grams let 'motorbike' land near 'motorcycle' without any outside model."""
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2), TruncatedSVD(DIM, random_state=0), Normalizer()).fit(texts)


def literal(vec):
    return "[" + ",".join(f"{x:.5f}" for x in vec) + "]"


# ---------------------------------------------------------------- places and time

def km(a, b):
    return math.hypot((a[0] - b[0]) * 111, (a[1] - b[1]) * 111 * math.cos(math.radians(a[0])))


def fold(s):
    """Upper-case letters only, with the digits a scanner confuses for letters turned back (1->I, 0->O, 5->S, 8->B)."""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z ]", " ", s.upper().translate(UNFOLD))).strip()


def city_of(a):
    if a["kind"] == "photo" and a["meta"].get("gps"):
        name, d = min(((n, km(a["meta"]["gps"], ll)) for n, ll in world.CITIES.items()), key=lambda x: x[1])
        return name if d < 60 else None
    if a["kind"] == "receipt":
        line = a["text"].split("\n")[1] if "\n" in a["text"] else ""
        return next((n for n in world.CITIES if n in line), None)
    return None


def moments(arts):
    """Group records into moments: a run with no two-hour silence and no jump of more than 600 m. -> [[index, ...]]"""
    out, cur, end, loc = [], [], None, None
    for i in sorted((i for i, a in enumerate(arts) if a["kind"] in MOMENT_KINDS), key=lambda i: arts[i]["at"]):
        a, g = arts[i], arts[i]["meta"].get("gps")
        if cur and (a["at"] - end > GAP or (g and loc and km(g, loc) > SPLIT_KM)):
            out.append(cur)
            cur, end, loc = [], None, None
        cur.append(i)
        end = max(end or a["at"], a.get("ends_at") or a["at"])
        loc = g or loc
    return out + ([cur] if cur else [])


def by_day(arts):
    """Baseline for moments: everything on the same calendar day is one event."""
    days = collections.defaultdict(list)
    for i, a in enumerate(arts):
        if a["kind"] in MOMENT_KINDS:
            days[a["at"].date()].append(i)
    return list(days.values())


def trips(arts):
    """Runs of consecutive days whose located records are in one city away from home. -> [{city, start, end}]"""
    seen = collections.defaultdict(collections.Counter)
    for a in arts:
        c = city_of(a)
        if c and c != world.HOME:
            seen[a["at"].date()][c] += 1
    out = []
    for day in sorted(seen):
        city = seen[day].most_common(1)[0][0]
        if out and out[-1]["city"] == city and (day - out[-1]["end"]).days <= 1:
            out[-1]["end"] = day
        else:
            out.append({"city": city, "start": day, "end": day})
    return out


def places(arts):
    """Restaurants, from receipts: the merchant line is folded so 'BELLAV1TA' and 'BELLAVITA' are one place.
    Cuisine is a vote over a small lexicon of dish and name words. -> ({key: place}, {receipt index: key})"""
    found, of = {}, {}
    for i, a in enumerate(arts):
        if a["kind"] == "receipt":
            first = a["text"].split("\n")[0]
            key = (city_of(a), fold(first))
            p = found.setdefault(key, {"city": key[0], "raw": collections.Counter(), "text": []})
            p["raw"][first] += 1
            p["text"].append(fold(a["text"]))
            of[i] = key
    for key, p in found.items():
        clean = [n for n in p["raw"] if not re.search(r"\d", n)]
        p["name"] = (max(clean, key=p["raw"].get) if clean else key[1]).title()
        blob = " ".join(p["text"])
        votes = {c: sum(blob.count(fold(w)) for w in pre + dishes) for c, (_, pre, _, dishes) in world.CUISINES.items()}
        p["cuisine"] = max(votes, key=votes.get) if max(votes.values()) else None
        p["receipts"] = sum(p["raw"].values())
    return found, of


def receipt_total(text):
    m = re.search(r"^TOTAL (\d+\.\d\d)$", text, re.M)
    return float(m[1]) if m else None


# ---------------------------------------------------------------- people

def people(arts, groups, naive=False):
    """Link mentions of people to contact cards. E-mail addresses and full names are certain; a bare first name or
    nickname is linked only if exactly one contact fits, or if the same moment names one of the candidates in full.
    -> ([(artifact index, contact index, confidence, how)], number of mentions left unresolved)"""
    contacts = [(i, a["meta"]) for i, a in enumerate(arts) if a["kind"] == "contact"]
    by_email = {m["email"]: i for i, m in contacts}
    by_name = {m["name"].lower(): i for i, m in contacts}
    short = collections.defaultdict(set)
    for i, m in contacts:
        short[m["name"].split()[0].lower()].add(i)
        if m.get("nick"):
            short[m["nick"].lower()].add(i)
    out, weak = [(i, i, 1.0, "contact card") for i, _ in contacts], []
    for i, a in enumerate(arts):
        if a["kind"] == "contact":
            continue
        text = " ".join([a["text"]] + list(a["meta"].get("attendees", []))).lower()
        strong = {by_email[e]: "e-mail address" for e in by_email if e in text} | {by_name[n]: "full name" for n in by_name if n in text and by_name[n] not in {by_email[e] for e in by_email if e in text}}
        out += [(i, c, 1.0 if how == "e-mail address" else 0.98, how) for c, how in strong.items()]
        words = set(re.findall(r"[a-z]+", (a["meta"].get("from") or "").lower())) if a["kind"] == "message" else set()
        for w in words & set(short):
            if not short[w] & set(strong):
                weak.append((i, w))
    moment_of = {i: k for k, g in enumerate(groups) for i in g}
    named = collections.defaultdict(set)
    for i, c, conf, _ in out:
        if conf >= 0.9 and i in moment_of:
            named[moment_of[i]].add(c)
    unresolved = 0
    for i, w in weak:
        cands = short[w]
        here = cands & named.get(moment_of.get(i), set())
        if len(cands) == 1:
            out.append((i, next(iter(cands)), 0.8, "first name, only one contact fits"))
        elif len(here) == 1:
            out.append((i, next(iter(here)), 0.7, "first name, settled by the calendar entry of the same moment"))
        elif naive:                                      # baseline: a shared first name always means the first such contact
            out.append((i, min(cands), 0.5, "first name, first contact with it"))
        else:
            unresolved += 1
    return out, unresolved


# ---------------------------------------------------------------- questions

def parse(question):
    """Pull the constraints out of a question with rules (no language model): year, city, cuisine, a photographed
    object, and what is being asked for."""
    q = question.strip()
    year = re.search(r"\b(20\d\d)\b", q)
    obj = re.search(r"(?:photographed|took a (?:picture|photo) of|(?:picture|photo) of)\s+(?:an?|the|my)?\s*([a-z][a-z ]*?)(?=\s+(?:outside|in|at|then|and|who|near|on)\b|[,.?!]|$)", q, re.I)
    low = q.lower()
    return {"year": int(year[1]) if year else None, "city": next((c for c in world.CITIES if c.lower() in low), None), "cuisine": next((c for c in world.CUISINES if c.lower() in low), None),
            "object": obj[1].strip().lower() if obj else None, "wants_receipt": bool(re.search(r"receipt|cost|paid|spend|spent|bill", low)), "wants_people": bool(re.search(r"who\b|with\b", low))}
