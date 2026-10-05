"""Unit tests for the indexing rules. No database."""
import datetime

import numpy as np
import pytest

from mem import engine, world

UTC = datetime.UTC
T = lambda h, m=0, d=1: datetime.datetime(2024, 5, d, h, m, tzinfo=UTC)      # noqa: E731
A = lambda kind, at, text="", ends=None, **meta: {"kind": kind, "at": at, "ends_at": ends, "text": text, "meta": meta}      # noqa: E731
CONTACTS = [A("contact", T(0), "", name="Alex Rivera", email="alex.rivera@mail.example", nick=None), A("contact", T(0), "", name="Alex Chen", email="alex.chen@mail.example", nick=None),
            A("contact", T(0), "", name="Priya Shah", email="priya.shah@mail.example", nick="Pri")]


def test_fold_undoes_scanner_confusions():
    assert engine.fold("OSTERIA BELLAV1TA") == engine.fold("Osteria Bellavita") == "OSTERIA BELLAVITA"
    assert engine.fold("2 T1RAMI5U 14.00") == "TIRAMISU I OO"                    # look-alike digits become letters everywhere, prices included: fold is for matching names only


@pytest.mark.parametrize("q", [x.format(cuisine="Italian", year=2024, city="Chicago", object="red motorcycle", an="an") for x in world.QUESTIONS])
def test_question_parser_handles_each_phrasing(q):
    p = engine.parse(q)
    assert (p["year"], p["city"], p["cuisine"], p["object"]) == (2024, "Chicago", "Italian", "red motorcycle") and p["wants_receipt"] and p["wants_people"]


def test_parser_leaves_out_what_was_not_said():
    p = engine.parse("anything from Lisbon?")
    assert p == {"year": None, "city": "Lisbon", "cuisine": None, "object": None, "wants_receipt": False, "wants_people": False}


def test_moments_split_on_silence_and_on_distance_and_respect_calendar_length():
    arts = [A("photo", T(14), gps=[41.88, -87.63]), A("photo", T(14, 20), gps=[41.8801, -87.6301]),
            A("photo", T(14, 40), gps=[41.90, -87.63]),                           # 2 km away: a new moment although only 20 minutes later
            A("calendar", T(19), "Dinner", ends=T(20, 50)), A("receipt", T(22, 30), "X\nChicago IL"),        # 100 minutes after the entry ends: same moment
            A("message", T(9, d=2), "hi"), A("email", T(9, 5, d=2), "newsletter")]                          # e-mail never forms a moment
    assert engine.moments(arts) == [[0, 1], [2], [3, 4], [5]]
    assert sorted(map(sorted, engine.by_day(arts))) == [[0, 1, 2, 3, 4], [5]]


def test_trips_are_runs_of_days_away_from_home():
    day = lambda d, city: A("receipt", T(20, d=d), f"X\n1 Main St, {city} XX")      # noqa: E731
    arts = [day(1, "Tampa"), day(3, "Chicago"), day(4, "Chicago"), day(5, "Chicago"), day(9, "Chicago"), day(10, "Denver")]
    got = [(t["city"], t["start"].day, t["end"].day) for t in engine.trips(arts)]
    assert got == [("Chicago", 3, 5), ("Chicago", 9, 9), ("Denver", 10, 10)]


def test_a_shared_first_name_is_only_linked_when_the_moment_settles_it():
    arts = CONTACTS + [A("calendar", T(19), "Dinner", ends=T(21), attendees=["alex.chen@mail.example"]), A("message", T(18, 50), "see you there", **{"from": "Alex"}),
                       A("message", T(9, d=3), "lunch?", **{"from": "Alex"}), A("message", T(9, 5, d=3), "yes", **{"from": "Pri"})]
    links, unresolved = engine.people(arts, engine.moments(arts))
    got = {(i, c): how for i, c, _, how in links if i >= 3}
    assert got[3, 1] == "e-mail address" and got[4, 1].startswith("first name, settled") and (6, 2) in got and unresolved == 1
    assert not any(i == 5 for i, _ in got)                                        # "Alex" alone, nothing else to go on: left unlinked
    naive, _ = engine.people(arts, engine.moments(arts), naive=True)
    assert (5, 0) in {(i, c) for i, c, _, _ in naive}


def test_receipts_with_scanner_errors_are_one_place_with_a_cuisine():
    r = lambda name: A("receipt", T(21), f"{name}\n412 W Kinzie St, Chicago IL\n1 T1RAMISU 12.00\n2 RISOTTO FUNGHI 40.00\nTOTAL 66.56")      # noqa: E731
    found, of = engine.places([r("OSTERIA BELLAVITA"), r("0STERIA BELLAV1TA"), r("OSTERIA BELLAVITA"), A("receipt", T(21), "KURO RAMEN\n9 Pine St, Seattle WA\n1 GYOZA 9.00\nTOTAL 11.52")])
    assert len(found) == 2 and of[0] == of[1] == of[2] != of[3]
    place = found[of[0]]
    assert (place["name"], place["city"], place["cuisine"], place["receipts"]) == ("Osteria Bellavita", "Chicago", "Italian", 3)
    assert engine.receipt_total("A\nSUBTOTAL 52.00\nTOTAL 66.56\nVISA") == 66.56 and engine.receipt_total("no total here") is None


def test_embedder_puts_a_synonym_nearer_than_an_unrelated_photo():
    docs = [engine.doc(a) for a in world.build(11)["artifacts"]]
    m = engine.embedder(docs)
    q, bike, other = m.transform(["photo red motorcycle", "photo red motorbike street parked", "photo wine glass candle burrata"])
    assert q @ bike > q @ other + 0.2 and np.allclose(np.linalg.norm([q, bike, other], axis=1), 1) and len(q) == engine.DIM
    assert np.allclose(m.transform(["photo red motorcycle"])[0], q)
