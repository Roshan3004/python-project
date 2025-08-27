from typing import List, Dict, Any
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure

class MongoStore:
    def __init__(self, uri: str, db: str, collection: str):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        try:
            self.client.admin.command('ping')
        except ConnectionFailure:
            raise RuntimeError("Cannot connect to MongoDB at URI: %s" % uri)
        self.col = self.client[db][collection]
        self.col.create_index([("period_id", ASCENDING)], unique=True)

    def upsert_many(self, rows: List[Dict[str, Any]]):
        ops = []
        for r in rows:
            ops.append({
                "update_one": {
                    "filter": {"period_id": r["period_id"]},
                    "update": {"$set": r},
                    "upsert": True
                }
            })
        if ops:
            self.col.bulk_write(ops)

    def latest_n(self, n: int = 1000) -> List[Dict[str, Any]]:
        cur = self.col.find().sort("period_id", -1).limit(n)
        return list(cur)[::-1]  # ascending by time
