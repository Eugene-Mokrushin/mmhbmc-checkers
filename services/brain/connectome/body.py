import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

from connectome.carve import clustered
from paths import DATA_DIR

WHERE = "https://raw.githubusercontent.com/TuragaLab/flybody/main/flybody/fruitfly/assets/{name}"
MODEL = "fruitfly.xml"
STORE = DATA_DIR / "flybody.npz"
CELLS = 160  # how finely the body is kept; it is a faint outline, not a specimen
UNIT = 1e7  # the model measures in centimetres, the connectome in nanometres


def turn(quat: str | None) -> np.ndarray:
    if not quat:
        return np.eye(3)
    w, x, y, z = (float(v) for v in quat.split())
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def shift(node) -> np.ndarray:
    return np.array([float(v) for v in (node.get("pos") or "0 0 0").split()])


def places(root) -> list[tuple[str, np.ndarray, np.ndarray]]:
    # every mesh in the model, with where it sits once the joints are followed home
    files = {mesh.get("name"): mesh.get("file") for mesh in root.find("asset").findall("mesh")}
    out: list[tuple[str, np.ndarray, np.ndarray]] = []

    def walk(body, rot, off):
        here, base = rot @ turn(body.get("quat")), off + rot @ shift(body)
        for geom in body.findall("geom"):
            if geom.get("mesh"):
                out.append((files[geom.get("mesh")], here @ turn(geom.get("quat")), base + here @ shift(geom)))
        for child in body.findall("body"):
            walk(child, here, base)

    for body in root.find("worldbody").findall("body"):
        walk(body, np.eye(3), np.zeros(3))
    return out


def fetch(name: str) -> str:
    kept = DATA_DIR / "flybody" / name
    if not kept.exists():
        kept.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(WHERE.format(name=name), timeout=120) as reply:
            kept.write_bytes(reply.read())
    return kept.read_text()


def obj(text: str) -> tuple[np.ndarray, np.ndarray]:
    points, faces = [], []
    for line in text.splitlines():
        if line.startswith("v "):
            points.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            corner = [int(part.split("/")[0]) - 1 for part in line.split()[1:]]
            faces += [[corner[0], corner[i], corner[i + 1]] for i in range(1, len(corner) - 1)]
    return np.array(points, dtype=np.float64), np.array(faces, dtype=np.int64)


def facing(points: np.ndarray) -> np.ndarray:
    # the model looks along +x with +z up; the connectome has x across the head, y down
    # it and z from front to back
    return np.stack([points[:, 1], -points[:, 2], -points[:, 0]], axis=1)


def main() -> None:
    # The fly the brain belongs to: the whole-body model published with flybody
    # (Vaxenburg et al., Apache 2.0), which was traced from a micro-CT scan of a real
    # fly. Kept about the middle of its head, in the connectome's own nanometres.
    root = ET.fromstring(fetch(MODEL))
    scale = float(root.find("default").find("mesh").get("scale").split()[0])
    every, corners, head, at = [], [], [], 0
    for name, rot, off in places(root):
        points, faces = obj(fetch(name))
        if not len(points):
            continue
        put = points * scale @ rot.T + off
        every.append(put)
        corners.append(faces + at)
        at += len(points)
        if name.startswith("head"):
            head.append(put)
    points, faces = np.concatenate(every), np.concatenate(corners)
    middle = np.concatenate(head)
    middle = (middle.min(axis=0) + middle.max(axis=0)) / 2
    points, faces = clustered(facing((points - middle) * UNIT), faces, CELLS)
    np.savez_compressed(STORE, points=points.astype(np.float32), faces=faces.astype(np.int32))
    print(f"{len(points)} points, {len(faces)} faces, {np.round((points.max(axis=0) - points.min(axis=0)) / 1000)} um across")


if __name__ == "__main__":
    main()
