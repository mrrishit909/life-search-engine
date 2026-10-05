"""A deterministic synthetic personal archive: three years of one invented person's photos, receipts, e-mail, calendar
and messages. Nobody here exists. Every record carries the truth it was generated from (which real-world event, which
people, which restaurant), so each indexing step can be scored.

There are no image files. A "photo" is the record an on-device image tagger would produce: time, GPS and a few tags."""
import datetime
import functools
import hashlib
import random

UTC = datetime.UTC
HOME = "Tampa"
CITIES = {"Tampa": (27.95, -82.46), "Chicago": (41.88, -87.63), "Denver": (39.74, -104.99), "Lisbon": (38.72, -9.14), "Seattle": (47.61, -122.33), "Austin": (30.27, -97.74),
          "New York": (40.71, -74.01), "Boston": (42.36, -71.06), "Montreal": (45.50, -73.57), "Mexico City": (19.43, -99.13)}
REGION = {"Tampa": "FL", "Chicago": "IL", "Denver": "CO", "Lisbon": "PT", "Seattle": "WA", "Austin": "TX", "New York": "NY", "Boston": "MA", "Montreal": "QC", "Mexico City": "MX"}
# cuisine -> (name pattern, proper names, dishes)
CUISINES = {
    "Italian": ("{p} {n}", ["Osteria", "Trattoria", "Ristorante"], ["Bellavita", "Fiorella", "Da Marco", "Il Forno", "Lucca", "Nonna Rosa", "Vicolo", "Portofino"], ["tagliatelle ragu", "risotto funghi", "tiramisu", "burrata", "cacio e pepe", "negroni"]),
    "Thai": ("{n} {p}", ["Thai", "Thai Kitchen"], ["Baan Suan", "Siam Garden", "Lotus", "Chiang Mai", "Sukhothai", "Krua"], ["pad thai", "green curry", "tom yum", "mango sticky rice", "papaya salad", "thai iced tea"]),
    "Mexican": ("{p} {n}", ["Taqueria", "Cantina", "Casa"], ["El Farol", "La Paloma", "Oaxaca", "Don Lupe", "Azul", "Mirasol"], ["tacos al pastor", "guacamole", "mole poblano", "elote", "carnitas", "margarita"]),
    "Japanese": ("{n} {p}", ["Sushi", "Izakaya", "Ramen"], ["Hoshi", "Kaze", "Tanuki", "Miyabi", "Sora", "Kuro"], ["tonkotsu ramen", "nigiri set", "gyoza", "edamame", "karaage", "sake flight"]),
    "French": ("{p} {n}", ["Bistro", "Brasserie", "Chez"], ["Margaux", "Le Coin", "Colette", "Lumiere", "Petit Jardin", "Odile"], ["steak frites", "onion soup", "creme brulee", "moules", "duck confit", "kir royale"]),
    "Indian": ("{n} {p}", ["Tandoor", "Indian Kitchen", "Curry House"], ["Saffron", "Jaipur", "Masala", "Chutney", "Bombay", "Kashi"], ["butter chicken", "garlic naan", "saag paneer", "lamb biryani", "samosa", "mango lassi"])}
STREETS = ["Kinzie St", "Alder Ave", "Rua Nova", "Pine St", "Congress Ave", "Mercer St", "Hanover St", "Rue Clark", "Calle Orizaba", "Bay Blvd", "Larimer St", "Maple Ave"]
FIRST = ["Priya", "Jonathan", "Alex", "Alex", "Mei", "Tomas", "Hannah", "Kwame", "Sofia", "Daniel", "Aisha", "Lucas", "Nora", "Ravi"]
LAST = ["Shah", "Bergstrom", "Rivera", "Chen", "Tanaka", "Novak", "Weiss", "Mensah", "Costa", "Okoro", "Rahman", "Moreau", "Lindqvist", "Iyer"]
NICK = {"Priya": "Pri", "Jonathan": "Jon", "Daniel": "Dan", "Hannah": "Han", "Tomas": "Tom"}
COLORS = ["red", "blue", "yellow", "green", "white", "black", "orange"]
THINGS = [("motorcycle", "motorbike"), ("bicycle", "bike"), ("tram", "streetcar"), ("door", "doorway"), ("dog", "puppy"), ("umbrella", "parasol"), ("car", "automobile"), ("mural", "wall painting"), ("boat", "sailboat")]
DEMO_OBJECTS = [("red", "motorcycle"), ("red", "car"), ("black", "motorcycle")]
SCENES = ["skyline", "river", "museum", "park", "bridge", "market", "cathedral", "harbor", "street", "rooftop", "sunset", "crowd", "statue", "garden"]
HOME_SCENES = ["beach", "cat", "kitchen", "garden", "sunset", "pier", "desk", "living room", "pelican", "palm tree"]
FOOD_TAGS = ["plate", "table", "wine glass", "candle", "menu", "dessert"]
NEWS = ["Your weekly digest: 12 new stories", "Statement available for your account ending 4417", "Sale ends Sunday: 30% off everything", "Your package has shipped", "Reminder: update your password",
        "New sign-in to your account", "This week in the neighborhood", "Your subscription renews soon"]
LATE = ["running 10 min late, see you at {r}", "table is booked at {r} for {h}", "just parked, walking over to {r}", "is it {r} tonight?", "got us a table by the window at {r}"]
OCR = {"I": "1", "O": "0", "S": "5", "B": "8"}


def checksum(kind, uri, text):
    return hashlib.sha256(f"{kind}|{uri}|{text}".encode()).hexdigest()


def near(r, latlon, km):
    return round(latlon[0] + r.uniform(-km, km) / 111, 5), round(latlon[1] + r.uniform(-km, km) / 90, 5)


def ocr(r, s, rate=0.04):
    return "".join(OCR[ch] if ch in OCR and r.random() < rate else ch for ch in s)


@functools.lru_cache(maxsize=8)
def build(seed=11, years=(2022, 2023, 2024), n_trips=12):
    r = random.Random(seed)
    people = [{"name": f"{f} {la}", "first": f, "email": f"{f}.{la}@mail.example".lower(), "nick": NICK.get(f)} for f, la in zip(FIRST, LAST)]
    arts, events, restaurants, n = [], [], {}, [0]

    def restaurant(city):
        """Reuse a restaurant in this city half the time; otherwise open a new one."""
        have = [k for k in restaurants if k[0] == city]
        if have and r.random() < (0.6 if city == HOME else 0.15):
            return restaurants[r.choice(have)]
        for _ in range(50):
            cuisine = r.choice(list(CUISINES))
            pat, prefixes, names, _ = CUISINES[cuisine]
            name = pat.format(p=r.choice(prefixes), n=r.choice(names))
            if (city, name) not in restaurants:
                break
        restaurants[city, name] = {"name": name, "city": city, "cuisine": cuisine, "gps": near(r, CITIES[city], 4), "address": f"{r.randrange(100, 999)} {r.choice(STREETS)}, {city} {REGION[city]}"}
        return restaurants[city, name]

    def add(kind, source, at, text, meta=None, ends=None, **truth):
        n[0] += 1
        uri = f"{source}/{at:%Y}/{kind}-{n[0]:05d}"
        arts.append({"kind": kind, "source": source, "uri": uri, "at": at, "ends_at": ends, "text": text, "meta": meta or {}, "checksum": checksum(kind, uri, text), "truth": truth})

    def photo(at, gps, tags, **truth):
        tags = list(tags)
        if r.random() < 0.1:
            tags[r.randrange(len(tags))] = r.choice(SCENES)          # the tagger gets one wrong now and then
        add("photo", "photos-export", at, " ".join(tags), {"gps": list(gps), "tags": tags}, **truth)

    def meal(day, hour, rest, who, ev, obj=None, calendar=True):
        """One meal out: maybe a calendar entry and a message beforehand, photos during, a receipt at the end."""
        t0 = datetime.datetime(day.year, day.month, day.day, hour, 0, tzinfo=UTC) + datetime.timedelta(minutes=r.randrange(0, 30))
        base = {"event": ev, "place": (rest["city"], rest["name"])}
        if who and calendar:
            add("calendar", "calendar-export", t0, f"Dinner at {rest['name']}" if hour >= 17 else f"Lunch at {rest['name']}", {"attendees": [people[p]["email"] for p in who], "location": rest["address"]},
                ends=t0 + datetime.timedelta(minutes=110), people=list(who), **base)
        if who and r.random() < 0.8:
            p = r.choice(who)
            add("message", "messages-export", t0 - datetime.timedelta(minutes=r.randrange(8, 25)), r.choice(LATE).format(r=rest["name"], h=f"{hour % 12}pm"), {"from": people[p]["nick"] if people[p]["nick"] and r.random() < 0.5 else people[p]["first"]},
                people=[p], **base)
        if obj:
            color, (thing, alt) = obj
            photo(t0 - datetime.timedelta(minutes=r.randrange(2, 9)), near(r, rest["gps"], 0.04), [color, r.choice([thing, alt]), "street", r.choice(["evening", "parked", "sidewalk"])], object=f"{color} {thing}", **base)
        for k in range(r.randrange(1 if obj else 0, 4)):
            photo(t0 + datetime.timedelta(minutes=20 + 22 * k + r.randrange(10)), near(r, rest["gps"], 0.03), r.sample(FOOD_TAGS, 2) + r.sample(CUISINES[rest["cuisine"]][3], 1), **base)
        items = [(r.randint(1, 2), d, round(r.uniform(9, 34), 2)) for d in r.sample(CUISINES[rest["cuisine"]][3], r.randint(3, 5))]
        sub = sum(q * p for q, _, p in items)
        total = round(sub * 1.28, 2)
        at = t0 + datetime.timedelta(minutes=r.randrange(95, 125))
        body = "\n".join([ocr(r, rest["name"].upper()), rest["address"], f"{at:%m/%d/%Y %H:%M}"] + [ocr(r, f"{q} {d.upper()}") + f" {q * p:.2f}" for q, d, p in items]
                         + [f"SUBTOTAL {sub:.2f}", f"TAX+TIP {total - sub:.2f}", f"TOTAL {total:.2f}", "VISA ****4417"])
        add("receipt", "receipts-scan", at, body, {"total": total}, **base, cuisine=rest["cuisine"])

    # ---- contacts
    for i, p in enumerate(people):
        add("contact", "contacts-export", datetime.datetime(years[0], 1, 1, tzinfo=UTC), f"{p['name']} {p['email']}" + (f" ({p['nick']})" if p["nick"] else ""), {"name": p["name"], "email": p["email"], "nick": p["nick"]}, people=[i])

    # ---- trips: (city, year) is unique; Chicago is visited in 2023 and in 2024
    plan = [("Chicago", 2023), ("Chicago", 2024)]
    while len(plan) < n_trips:
        c = (r.choice([c for c in CITIES if c != HOME]), r.choice(years))
        if c not in plan:
            plan.append(c)
    away, trips = set(), []
    for city, year in plan:
        for _ in range(100):
            start = datetime.date(year, 1, 1) + datetime.timedelta(days=r.randrange(5, 350))
            days = [start + datetime.timedelta(days=k) for k in range(r.randint(3, 5))]
            if not any(d + datetime.timedelta(days=k) in away for d in days for k in (-2, -1, 0, 1, 2)):
                break
        away.update(days)
        who = r.sample(range(len(people)), r.randint(1, 3))
        trip = {"city": city, "start": days[0], "end": days[-1], "people": who, "dinners": []}
        trips.append(trip)
        booked = datetime.datetime(year, days[0].month, days[0].day, 9, tzinfo=UTC) - datetime.timedelta(days=r.randint(15, 40), hours=r.randrange(8))
        add("email", "mail-export", booked, f"Your trip to {city} is confirmed. Departing {days[0]:%Y-%m-%d} from {HOME}, returning {days[-1]:%Y-%m-%d}. Confirmation {r.randrange(10**5, 10**6)}.", {"from": "bookings@airline.example", "subject": f"Itinerary: {HOME} to {city}"}, trip=len(trips) - 1)
        objects = r.sample([(c, t) for c in COLORS for t in THINGS if (c, t[0]) not in DEMO_OBJECTS], len(days) * 3)
        demo_day = 1 if (city, year) == ("Chicago", 2024) else None      # the question the demo asks is planted here, with two near-misses the same day
        if demo_day:
            objects[3:6] = [(c, next(x for x in THINGS if x[0] == t)) for c, t in DEMO_OBJECTS]
        for k, day in enumerate(days):
            if r.random() < 0.6:
                events.append({"kind": "lunch", "trip": len(trips) - 1})
                meal(day, 12, restaurant(city), r.sample(who, r.randint(0, len(who))), len(events) - 1, calendar=False)
            events.append({"kind": "sightseeing", "trip": len(trips) - 1})
            spot = near(r, CITIES[city], 5)
            for j in range(r.randint(3, 6)):
                at = datetime.datetime(day.year, day.month, day.day, 14, tzinfo=UTC) + datetime.timedelta(minutes=25 * j + r.randrange(15))
                c2, (t2, a2) = objects[k * 3 + 1 + j % 2]       # other coloured things seen that day: same city, not the dinner's
                photo(at, near(r, spot, 0.2), r.sample(SCENES, 2) + ([c2, r.choice([t2, a2])] if j < 2 else [r.choice(SCENES)]), event=len(events) - 1)
            rest = restaurant(city)
            if k == demo_day:
                rest = restaurants.setdefault((city, "Osteria Bellavita"), {"name": "Osteria Bellavita", "city": city, "cuisine": "Italian", "gps": near(r, CITIES[city], 4), "address": f"412 W Kinzie St, {city} IL"})
            with_ = r.sample(who, r.randint(1, len(who)))
            events.append({"kind": "dinner", "trip": len(trips) - 1, "city": city, "year": year, "restaurant": rest["name"], "cuisine": rest["cuisine"], "people": sorted(with_), "object": f"{objects[k * 3][0]} {objects[k * 3][1][0]}"})
            trip["dinners"].append(len(events) - 1)
            meal(day, 19, rest, with_, len(events) - 1, obj=objects[k * 3], calendar=k == demo_day or r.random() < 0.75)

    # ---- life at home
    day = datetime.date(years[0], 1, 3)
    while day.year <= years[-1]:
        if day not in away:
            if r.random() < 0.2:
                events.append({"kind": "home dinner"})
                meal(day, 19, restaurant(HOME), r.sample(range(len(people)), r.randint(1, 2)), len(events) - 1, calendar=r.random() < 0.7)
            if r.random() < 0.3:
                events.append({"kind": "snapshot"})
                at = datetime.datetime(day.year, day.month, day.day, r.randrange(8, 17), r.randrange(60), tzinfo=UTC)
                for j in range(r.randint(1, 3)):
                    photo(at + datetime.timedelta(minutes=3 * j), near(r, CITIES[HOME], 8), r.sample(HOME_SCENES, 3), event=len(events) - 1)
            if r.random() < 0.4:
                at = datetime.datetime(day.year, day.month, day.day, r.randrange(6, 22), r.randrange(60), tzinfo=UTC)
                add("email", "mail-export", at, r.choice(NEWS), {"from": "no-reply@service.example", "subject": "Update"})
        day += datetime.timedelta(days=1)
    arts.sort(key=lambda a: a["at"])
    return {"seed": seed, "people": people, "artifacts": arts, "events": events, "trips": trips, "restaurants": list(restaurants.values())}


QUESTIONS = ["Find the {cuisine} restaurant from my {year} {city} trip where I photographed a {object}, then show the receipt and who I was with.",
             "Which {cuisine} place did I eat at in {city} in {year}, the one where I took a picture of a {object}? Who was there and what did it cost?",
             "In {year} I was in {city} and photographed a {object} outside {an} {cuisine} restaurant. Show me the receipt and who I was with."]


def questions(w):
    """One multi-hop question per trip dinner, with its true answer."""
    out = []
    for i, e in enumerate(w["events"]):
        if e["kind"] == "dinner":
            receipt = next(a for a in w["artifacts"] if a["kind"] == "receipt" and a["truth"]["event"] == i)
            out.append({"question": QUESTIONS[i % 3].format(an="an" if e["cuisine"][0] in "AEIOU" else "a", **e), "event": i, "restaurant": e["restaurant"], "receipt": receipt["checksum"], "people": {w["people"][p]["name"] for p in e["people"]}, "city": e["city"], "year": e["year"], "object": e["object"]})
    return out
