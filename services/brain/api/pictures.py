import asyncio
import hashlib

from fastapi import FastAPI, HTTPException, Response

from api.atlas import SKELETONS, Drawings, atlas, frame
from api.symmetry import Envelope
from connectome.coordinates import one_per_neuron, read_markers


def keep(blob: bytes) -> dict:
    # cached in the browser, but checked against its tag, so a rebuilt brain arrives
    return {"Cache-Control": "public, max-age=3600, must-revalidate", "ETag": hashlib.sha1(blob).hexdigest()[:16]}


def pictures(app: FastAPI, flies):
    # what the website needs to draw a brain: the places each of a fly's neurons runs
    # through. It is the same for every game, so it is worked out once and kept.
    held: dict = {"points": None, "frame": None, "drawn": None, "inside": None, "atlas": {}}

    async def anatomy():
        if held["points"] is None:
            held["points"] = await asyncio.to_thread(lambda: one_per_neuron(read_markers()))
            held["frame"] = await asyncio.to_thread(frame, held["points"])
            if SKELETONS.exists():
                held["drawn"] = await asyncio.to_thread(Drawings)
            places = held["drawn"].xyz if held["drawn"] is not None else held["points"].to_numpy(dtype=float)
            held["inside"] = await asyncio.to_thread(Envelope, places)
            if held["drawn"] is not None:
                await asyncio.to_thread(held["drawn"].trim, held["inside"])
        return held["points"], *held["frame"]

    @app.get("/atlas/{fly}")
    async def positions(fly: str) -> Response:
        stable = flies()
        if fly not in stable.flies:
            raise HTTPException(404, f"no fly called {fly}")
        if fly not in held["atlas"]:
            points, middle, spread = await anatomy()
            made = await asyncio.to_thread(atlas, stable.flies[fly].root_id, points, middle, spread, held["drawn"], held["inside"])
            held["atlas"][fly] = made
        return Response(held["atlas"][fly], media_type="application/octet-stream", headers=keep(held["atlas"][fly]))

    async def warm() -> None:
        # a brain takes a while to draw, so draw them all before anyone asks
        for name in list(flies().flies):
            await positions(name)

    return warm
