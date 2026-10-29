"""Ingestion orchestration: incremental refresh keyed on a content hash of each page."""
import json
import logging
import threading
import time

from .config import Settings
from .ingest import Fetcher, build_page, load_sources
from .store import KnowledgeStore

log = logging.getLogger(__name__)
_lock = threading.Lock()
status: dict = {"running": False, "last_run": None, "last_result": None}


def refresh(store: KnowledgeStore, s: Settings, force: bool = False) -> dict:
    """Re-fetch every source. Only pages whose cleaned text changed are re-embedded.
    A page that fails to fetch keeps its previous content (stale beats empty)."""
    if not _lock.acquire(blocking=False):
        return {"error": "refresh already running"}
    status["running"] = True
    result: dict = {"updated": [], "unchanged": [], "failed": {}, "removed": []}
    try:
        manifest_path = s.data_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        sources = load_sources(s.sources_file)
        fetcher = Fetcher(s.crawler_user_agent, s.crawl_delay_seconds)

        for src in sources:
            url = src["url"]
            try:
                page = build_page(url, src.get("page_type", "page"), fetcher.get(url))
                if not page.chunks:
                    raise ValueError("no extractable content")
                if not force and manifest.get(url) == page.content_hash and store.count():
                    result["unchanged"].append(url)
                    continue
                n = store.upsert_page(page)
                manifest[url] = page.content_hash
                result["updated"].append({"url": url, "chunks": n})
            except Exception as exc:  # noqa: BLE001 - keep going, report per page
                log.exception("Ingest failed for %s", url)
                result["failed"][url] = str(exc)

        wanted = {src["url"] for src in sources}
        for url in list(manifest):
            if url not in wanted:
                store.delete_url(url)
                manifest.pop(url)
                result["removed"].append(url)

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return result
    finally:
        status.update(running=False, last_run=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), last_result=result)
        _lock.release()
