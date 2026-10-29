import logging
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import pipeline
from .config import get_settings
from .db import Database
from .rag import RagService
from .store import KnowledgeStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")
settings = get_settings()
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app.state.store = KnowledgeStore(settings)
    app.state.db = Database(settings.data_dir / "bot.db")
    app.state.rag = RagService(settings, app.state.store, app.state.db)
    if app.state.store.count() == 0:  # first boot: build the index in the background
        threading.Thread(target=pipeline.refresh, args=(app.state.store, settings), daemon=True).start()
    yield


app = FastAPI(title="PlacidWay Knowledge Chatbot", lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins.split(","), allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    return resp


# --- rate limiting (per client IP, in-memory; fine for a single instance) ---------------------
_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request):
    ip = (request.headers.get("x-forwarded-for") or request.client.host or "?").split(",")[0].strip()
    now, q = time.monotonic(), _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= settings.rate_limit_per_minute:
        raise HTTPException(429, "Too many requests. Please wait a moment.")
    q.append(now)


def require_admin(x_admin_token: str = Header(default="")):
    if not settings.admin_token or not secrets.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(403, "Forbidden")


# --- schemas -----------------------------------------------------------------------------------
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=12)
    disclaimer_shown: bool = False


class LeadIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(max_length=200)
    message: str = Field(default="", max_length=1000)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            raise ValueError("invalid email")
        return v


# --- public API --------------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok", "chunks": app.state.store.count()}


@app.post("/api/chat", dependencies=[Depends(rate_limit)])
def chat(body: ChatIn):
    if app.state.store.count() == 0:
        raise HTTPException(503, "The knowledge base is still being built. Please try again in a minute.")
    try:
        r = app.state.rag.respond(body.message.strip(), [t.model_dump() for t in body.history], body.disclaimer_shown)
    except Exception:  # noqa: BLE001
        log.exception("chat failed")
        raise HTTPException(503, "The assistant is temporarily unavailable. Please try again shortly.")
    return {"answer": r.answer, "type": r.type, "sources": r.sources, "disclaimer": r.disclaimer, "offer_lead": r.offer_lead}


@app.post("/api/lead", dependencies=[Depends(rate_limit)])
def lead(body: LeadIn):
    app.state.db.add_lead(body.name, body.email, body.message)
    return {"ok": True, "contact_url": settings.contact_url}


# --- admin -------------------------------------------------------------------------------------
@app.post("/admin/refresh", dependencies=[Depends(require_admin)], status_code=202)
def admin_refresh(tasks: BackgroundTasks, force: bool = False):
    if pipeline.status["running"]:
        return JSONResponse({"detail": "refresh already running"}, status_code=409)
    tasks.add_task(pipeline.refresh, app.state.store, settings, force)
    return {"detail": "refresh started", "force": force}


@app.get("/admin/status", dependencies=[Depends(require_admin)])
def admin_status():
    return {**pipeline.status, "chunks": app.state.store.count()}


@app.get("/admin/unanswered", dependencies=[Depends(require_admin)])
def admin_unanswered():
    return app.state.db.rows("unanswered")


@app.get("/admin/leads", dependencies=[Depends(require_admin)])
def admin_leads():
    return app.state.db.rows("leads")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
