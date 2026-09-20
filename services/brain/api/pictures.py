import asyncio

from fastapi import FastAPI, HTTPException, Response

from api.atlas import atlas, frame, surface
from connectome.coordinates import one_per_neuron, read_markers

KEEP = {"Cache-Control": "public, max-age=604800"}


def pictures(app: FastAPI, flies) -> None:
    # what the website needs to draw a brain: where a fly's neurons sit, and the
    # outline of the brain they sit in. Both are the same for every game, so they are
    # worked out once and kept.
    held: dict = {"points": None, "frame": None, "atlas": {}, "mesh": None}

    async def anatomy():
        if held["points"] is None:
            held["points"] = await asyncio.to_thread(lambda: one_per_neuron(read_markers()))
            held["frame"] = await asyncio.to_thread(frame, held["points"])
        return held["points"], *held["frame"]

    @app.get("/atlas/{fly}")
    async def positions(fly: str) -> Response:
        stable = flies()
        if fly not in stable.flies:
            raise HTTPException(404, f"no fly called {fly}")
        if fly not in held["atlas"]:
            points, middle, spread = await anatomy()
            held["atlas"][fly] = await asyncio.to_thread(atlas, stable.flies[fly].root_id, points, middle, spread)
        return Response(held["atlas"][fly], media_type="application/octet-stream", headers=KEEP)

    @app.get("/mesh")
    async def mesh() -> Response:
        if held["mesh"] is None:
            _, middle, spread = await anatomy()
            held["mesh"] = await asyncio.to_thread(surface, middle, spread)
        return Response(held["mesh"], media_type="application/octet-stream", headers=KEEP)
