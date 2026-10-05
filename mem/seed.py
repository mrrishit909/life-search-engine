"""Tenants (one per archive owner) and demo tokens. The archive itself arrives through POST /v1/connectors/import.

    python -m mem.seed [--if-empty]
"""
import hashlib
import sys
import uuid

from core import db

TENANTS = {"maya": ("Maya Okafor's archive", 11, 12), "other": ("Second archive", 29, 4)}       # name, world seed, trips
ROLES = ["owner", "reader"]


def token(tenant, role):
    """Demo credentials for local use only."""
    return f"{tenant}-{role}-demo"


def main(if_empty=False):
    db.migrate()
    out = {}
    with db.tx() as c:
        if if_empty and c.execute("SELECT 1 FROM tenants LIMIT 1").fetchone():
            return print("already seeded")
        for key, (name, _, _) in TENANTS.items():
            t = uuid.uuid5(uuid.NAMESPACE_DNS, f"{key}.life-search.example")
            c.execute("INSERT INTO tenants (id, name) VALUES (%s,%s)", [t, name])
            for role in ROLES:
                c.execute("INSERT INTO api_tokens VALUES (%s,%s,%s,%s,%s)", [hashlib.sha256(token(key, role).encode()).hexdigest(), t, uuid.uuid5(t, role), f"{role} ({name})", role])
            out[key] = t
    print("demo tokens:", ", ".join(token(t, r) for t in TENANTS for r in ROLES))
    return out


if __name__ == "__main__":
    main("--if-empty" in sys.argv)
