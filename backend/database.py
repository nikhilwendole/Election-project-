"""
database.py — MongoDB Integration
"""

import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import os
import hashlib

load_dotenv()

MONGO_URI       = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME         = os.getenv("DB_NAME", "election_system")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "voters")

_client     = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client     = MongoClient(MONGO_URI)
        _collection = _client[DB_NAME][COLLECTION_NAME]
        _collection.create_index("voter_id", unique=True)
    return _collection


def init_db():
    try:
        col = get_collection()
        col.database.client.admin.command("ping")
        print("[DB] Connected to MongoDB Successfully!")
    except Exception as e:
        print(f"[DB] Connection FAILED: {e}")
        raise


def generate_voter_id(encoding: list) -> str:
    raw = str(encoding).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def save_voter(name: str, encoding: list, voter_number: str = "") -> dict:
    col      = get_collection()
    voter_id = generate_voter_id(encoding)

    document = {
        "voter_id"     : voter_id,
        "voter_name"   : name,
        "voter_number" : voter_number or f"VN-{voter_id[:8].upper()}",
        "face_encoding": encoding,
        "voted_at"     : datetime.utcnow(),
        "status"       : "voted",
    }

    col.insert_one(document)
    print(f"[DB] Voter saved — Name: {name} | ID: {voter_id}")
    return document


def get_all_voters() -> list:
    col    = get_collection()
    rows   = col.find({}, {"_id": 0})
    voters = []
    for row in rows:
        voters.append({
            "voter_id"    : row.get("voter_id", ""),
            "name"        : row.get("voter_name", ""),
            "voter_number": row.get("voter_number", ""),
            "encoding"    : row.get("face_encoding", []),
            "voted_at"    : row.get("voted_at", ""),
        })
    return voters


def list_voters():
    voters = get_all_voters()
    if not voters:
        print("\n[DB] No voters registered yet.")
        return

    print("\n" + "="*60)
    print("           REGISTERED VOTERS IN MONGODB")
    print("="*60)
    for i, v in enumerate(voters, 1):
        print(f"  {i}. Name      : {v['name']}")
        print(f"     Voter No  : {v['voter_number']}")
        print(f"     Voter ID  : {v['voter_id']}")
        print(f"     Voted At  : {v['voted_at']}")
        print("-"*60)
    print(f"  Total Voters: {len(voters)}")
    print("="*60)