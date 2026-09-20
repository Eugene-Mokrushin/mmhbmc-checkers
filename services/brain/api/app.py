import asyncio
import base64
import contextlib
import os
import time

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from api.pictures import pictures
from api.flies import Stable
from api.spikes import packed
from flycore.board import Position


class Ask(BaseModel):
    fly: str = Field(min_length=1, max_length=40)
    depth: int = Field(default=1, ge=1, le=4)
    position: list[int] = Field(min_length=4, max_length=4)
    spikes: bool = True


def build(stable: Stable | None = None) -> FastAPI:
    key = os.environ.get("BRAIN_KEY", "")

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        # loading the flies takes a while; do it before the first visitor arrives
        await asyncio.to_thread(stable_now)
        yield

    app = FastAPI(title="fly brain", lifespan=lifespan)
    app.state.stable = stable
    app.state.locks = {}

    def guard(given: str) -> None:
        if key and given != key:
            raise HTTPException(401, "who are you")

    def stable_now() -> Stable:
        if app.state.stable is None:
            app.state.stable = Stable()
            app.state.locks = {device: asyncio.Lock() for device in {f.device for f in app.state.stable.flies.values()}}
        return app.state.stable

    pictures(app, stable_now)

    @app.get("/health")
    async def health() -> dict:
        stable = app.state.stable
        return {"awake": stable is not None, **(stable.state() if stable else {})}

    @app.get("/flies")
    async def flies(x_brain_key: str = Header(default="")) -> dict:
        guard(x_brain_key)
        stable = stable_now()
        return {"flies": [{"name": name, "depths": [1, 2, 3]} for name in stable.flies]}

    @app.post("/choose")
    async def choose(ask: Ask, x_brain_key: str = Header(default="")) -> dict:
        guard(x_brain_key)
        stable = stable_now()
        if ask.fly not in stable.flies:
            raise HTTPException(404, f"no fly called {ask.fly}")
        lock = app.state.locks.setdefault(stable.flies[ask.fly].device, asyncio.Lock())
        started = time.monotonic()
        async with lock:
            move, score, frames = await asyncio.to_thread(stable.choose, ask.fly, ask.depth, Position(*ask.position))
            if move is None:
                return {"move": None, "score": score, "thinking_ms": 0}
            blob = await asyncio.to_thread(packed, frames) if ask.spikes else b""
        answer = {
            "move": {"origin": move.origin, "destination": move.destination, "captured": move.captured, "path": list(move.path), "promotes": move.promotes},
            "score": round(score, 4),
            "thinking_ms": int((time.monotonic() - started) * 1000),
        }
        if blob:
            answer["spikes"] = base64.b64encode(blob).decode()
        return answer

    return app
