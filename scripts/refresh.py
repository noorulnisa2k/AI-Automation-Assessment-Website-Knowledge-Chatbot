"""CLI refresh: python -m scripts.refresh [--force]"""
import argparse
import json

from app.config import get_settings
from app.pipeline import refresh
from app.store import KnowledgeStore

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-embed every page even if unchanged")
    s = get_settings()
    print(json.dumps(refresh(KnowledgeStore(s), s, ap.parse_args().force), indent=2))
