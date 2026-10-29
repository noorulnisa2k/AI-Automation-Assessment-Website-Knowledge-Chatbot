"""Fetch the whitelisted pages politely and turn them into structured chunks."""
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from urllib import robotparser
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

MAX_CHUNK = 1200
MIN_CHUNK = 400
NOISE_TAGS = ["script", "style", "noscript", "nav", "footer", "aside", "form", "iframe", "svg"]
NOISE_ATTR = re.compile(r"cookie|breadcrumb|sidebar|menu|popup|modal", re.I)
BLOCK_TAGS = ["h1", "h2", "h3", "h4", "p", "li", "table", "dt", "dd"]


@dataclass
class Chunk:
    section: str
    text: str


@dataclass
class Page:
    url: str
    title: str
    page_type: str
    content_hash: str
    chunks: list[Chunk] = field(default_factory=list)


def load_sources(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        sources = json.load(f)
    return [s for s in sources if s.get("url")]


class Fetcher:
    """Polite HTTP client: robots.txt aware, identifying User-Agent, ~1 request/second."""

    def __init__(self, user_agent: str, delay: float):
        self.ua, self.delay = user_agent, delay
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._robots: dict[str, robotparser.RobotFileParser] = {}
        self._last = 0.0

    def _throttle(self):
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _robots_for(self, url: str) -> robotparser.RobotFileParser:
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self._robots:
            rp = robotparser.RobotFileParser()
            self._throttle()
            try:
                r = self.session.get(f"{origin}/robots.txt", timeout=15)
                if r.status_code in (401, 403):
                    rp.disallow_all = True
                elif r.ok:
                    rp.parse(r.text.splitlines())
                else:
                    rp.allow_all = True
            except requests.RequestException as exc:
                log.warning("robots.txt unreachable (%s); assuming allowed", exc)
                rp.allow_all = True
            self._robots[origin] = rp
        return self._robots[origin]

    def get(self, url: str) -> str:
        if not self._robots_for(url).can_fetch(self.ua, url):
            raise PermissionError(f"robots.txt disallows {url}")
        self._throttle()
        r = self.session.get(url, timeout=25)
        r.raise_for_status()
        return r.text


def extract_blocks(html: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (title, [(kind, text)]) where kind is 'h' (heading) or 'b' (body block)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    for attr in ("class", "id"):
        for tag in soup.find_all(attrs={attr: NOISE_ATTR}):
            tag.decompose()

    h1 = soup.find("h1")
    title = (h1.get_text(" ", strip=True) if h1 else "") or (soup.title.get_text(strip=True) if soup.title else "")
    root = soup.find("main") or soup.find("article") or soup.body or soup

    blocks: list[tuple[str, str]] = []
    for el in root.find_all(BLOCK_TAGS):
        if el.name != "table" and el.find_parent("table"):
            continue
        if el.name == "li" and el.find("li"):
            continue
        if el.name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            text, kind = "\n".join(rows), "b"
        elif el.name in ("h1", "h2", "h3", "h4"):
            text, kind = el.get_text(" ", strip=True), "h"
        else:
            text, kind = el.get_text(" ", strip=True), "b"
            if el.name == "li":
                text = f"- {text}"
        if text and len(text) > 1:
            blocks.append((kind, text))
    return title, blocks


def _split(text: str) -> list[str]:
    """Never split a block unless it exceeds MAX_CHUNK; then split on line/sentence boundaries."""
    if len(text) <= MAX_CHUNK:
        return [text]
    parts, cur = [], ""
    for piece in re.split(r"(?<=[\n.!?])\s+", text):
        if len(cur) + len(piece) > MAX_CHUNK and cur:
            parts.append(cur.strip())
            cur = ""
        cur += piece + " "
    if cur.strip():
        parts.append(cur.strip())
    return parts


def chunk_blocks(blocks: list[tuple[str, str]]) -> list[Chunk]:
    """Heading-aware packing. Tables and lists stay intact; every chunk carries its heading."""
    chunks: list[Chunk] = []
    cur: list[str] = []
    size = 0
    section = cur_section = ""

    def flush():
        nonlocal cur, size
        body = "\n".join(cur).strip()
        if len(body.replace("#", "").strip()) > 40:
            chunks.append(Chunk(cur_section, body))
        cur, size = [], 0

    for kind, text in blocks:
        if kind == "h":
            section = text
            if size >= MIN_CHUNK:
                flush()
            if not cur:
                cur_section = section
            cur.append(f"## {text}")
            continue
        for piece in _split(text):
            if size + len(piece) > MAX_CHUNK and cur:
                flush()
                cur_section = section
                if section:
                    cur.append(f"## {section}")
            cur.append(piece)
            size += len(piece)
    flush()
    return chunks


def build_page(url: str, page_type: str, html: str) -> Page:
    title, blocks = extract_blocks(html)
    clean = "\n".join(t for _, t in blocks)
    if len(clean) < 300:
        log.warning("Very little text extracted from %s (%d chars) - page may be JS-rendered", url, len(clean))
    return Page(
        url=url,
        title=title or url,
        page_type=page_type,
        content_hash=hashlib.sha256(clean.encode()).hexdigest(),
        chunks=chunk_blocks(blocks),
    )
