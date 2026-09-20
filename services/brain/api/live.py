import asyncio

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from flycore.board import Position
from api.flies import spoken
from api.spikes import packed
from game.players import candidates

HELLO = "who are you"


def live(app: FastAPI, flies, key: str) -> None:
    # The fly thinking, as it thinks: every 5 ms of its brain time goes out as the
    # neurons that fired in it, and the move follows when the last frame is done.
    @app.websocket("/watch")
    async def watch(socket: WebSocket) -> None:
        await socket.accept()
        try:
            ask = await socket.receive_json()
            if key and ask.get("key") != key:
                await socket.close(code=4401, reason=HELLO)
                return
            await think(socket, flies(), ask)
        except WebSocketDisconnect:
            return

    async def think(socket: WebSocket, stable, ask: dict) -> None:
        fly = stable.flies.get(ask.get("fly", ""))
        if fly is None:
            await socket.send_json({"trouble": f"no fly called {ask.get('fly')}"})
            return
        options, after = candidates([Position(*ask["position"])])
        if not after:
            await socket.send_json({"move": None, "score": -1.0})
            return
        await socket.send_json({"boards": len(after), "neurons": fly.player.sim.n, "frame_ms": 5})
        counts, frames = await pour(socket, fly, after)
        move, score, chosen = stable.pick(fly, options[0], after, counts, ask.get("depth", 1))
        await socket.send_json({"move": spoken(move), "score": round(score, 4), "chosen": chosen, "spikes": len(packed(frames[chosen]))})

    async def pour(socket: WebSocket, fly, after) -> tuple:
        # the simulation runs in a thread, and each frame is handed over as it lands
        run = fly.watching(after)
        loop = asyncio.get_running_loop()
        while True:
            kind, payload = await loop.run_in_executor(None, lambda: next(run, ("end", None)))
            if kind == "end":
                return payload
            await socket.send_bytes(np.uint32(len(payload)).tobytes() + np.asarray(payload, dtype="<u4").tobytes())
