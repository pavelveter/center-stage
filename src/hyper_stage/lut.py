"""3D .cube LUT loading, baking and application."""

import glob
import os

import cv2
import numpy as np


def load_lut(path):
    """Parse a .cube 3D LUT into a (N,N,N,3) float32 array (axes r,g,b)."""
    if not path:
        return None
    if not os.path.exists(path):
        print(f"warning: LUT file not found: {path}")
        return None
    size = None
    data = []
    # .cube files are text: comments (#), optional TITLE/DOMAIN headers, a size
    # declaration, then size**3 RGB triples. Only the triples are consumed.
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("TITLE") or line.upper().startswith("DOMAIN_"):
                continue
            if line.upper().startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                continue
            if line.upper().startswith("LUT_1D_SIZE"):
                print("warning: 1D LUTs are not supported; ignoring")
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    data.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    pass
    if size is None or len(data) != size * size * size:
        print(f"warning: invalid LUT {path}: size={size} entries={len(data)}")
        return None
    # file order is R fastest -> C-order (b,g,r); transpose to (r,g,b)
    lut = np.array(data, dtype=np.float32).reshape(size, size, size, 3)
    lut = np.transpose(lut, (2, 1, 0, 3))
    return lut


def bake_lut(lut, grid=128):
    """Bake once at startup: (N,N,N,3) float (r,g,b) -> flat uint8 BGR table.

    Trilinear interpolation happens here, not per frame. At runtime only one
    take() on the precomputed 128^3 grid (7 bits per channel) with no float math.
    """
    if lut is None:
        return None
    assert grid == 128, "apply_lut indexing assumes 7 bits per channel"
    N = lut.shape[0]
    pos = np.linspace(0, N - 1, grid, dtype=np.float32)
    i0 = np.floor(pos).astype(np.int32)
    i1 = np.minimum(i0 + 1, N - 1)
    w = (pos - i0).astype(np.float32)
    o = lut.astype(np.float32)
    # separate per-axis linear interpolation = trilinear
    o = o[i0] * (1 - w)[:, None, None, None] + o[i1] * w[:, None, None, None]
    o = o[:, i0] * (1 - w)[None, :, None, None] + o[:, i1] * w[None, :, None, None]
    o = (
        o[:, :, i0] * (1 - w)[None, None, :, None]
        + o[:, :, i1] * w[None, None, :, None]
    )
    table = np.clip(o * 255.0 + 0.5, 0, 255).astype(np.uint8)
    table = table[..., ::-1]  # LUT yields RGB, frame and virtual camera are BGR
    return np.ascontiguousarray(table.reshape(-1, 3))


def apply_lut(frame_bgr, table, scale=8, detail=True):
    """Fast LUT application.

    Color is computed on a downscaled copy (1/scale), then one of two modes:
      detail=True  - the graded frame is upscaled and the luma-only detail
                     of the original is added back: sharpness is preserved
                     without leaking the original colors through the detail;
      detail=False - the graded frame itself is upscaled (faster, but softer).
    """
    if table is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    scale = max(2, int(scale))
    scale = min(scale, max(2, w // 4), max(2, h // 4))

    # cheap downscale: 2x2-box + bilinear (much faster than INTER_AREA)
    half = cv2.resize(frame_bgr, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    small = cv2.resize(half, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)

    # integer index into the 128^3 table: (r<<14)|(g<<7)|b, frame is BGR
    c = (small >> 1).astype(np.int32)
    idx = (c[..., 2] << 14) | (c[..., 1] << 7) | c[..., 0]
    graded = table.take(idx, axis=0)

    if not detail:
        return cv2.resize(graded, (w, h), interpolation=cv2.INTER_LINEAR)

    # Sharpness detail with the chroma stripped. The naive form
    # (frame + upscaled correction) keeps the full-res chroma, so a B&W
    # grade leaks the original colors back at edges and in sensor noise.
    # Adding only the luma detail to the upscaled graded image preserves
    # the sharpness this mode exists for without re-tinting the grade.
    # cvtColor's gray conversion uses the same Rec.601 weights as a manual
    # 0.299/0.587/0.114 transform but is several times faster.
    luma = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    luma_small_up = cv2.resize(
        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (w, h), interpolation=cv2.INTER_LINEAR
    )
    # Signed luma correction stored as uint8 with offset 128 (range +-127);
    # one saturating op replaces the int16 subtract/offset/clip/astype chain.
    delta8 = cv2.addWeighted(luma, 1.0, luma_small_up, -1.0, 128.0)
    graded_up = cv2.resize(graded, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(
        graded_up, 1.0, cv2.cvtColor(delta8, cv2.COLOR_GRAY2BGR), 1.0, -128.0
    )


def resolve_lut_path(path, base_dir=None):
    """Resolve a LUT path relative to the config dir, cwd, or as-is.

    Returns None for an empty path; otherwise the first existing candidate or
    the original path so the caller can report a clear "not found" warning.
    """
    if not path:
        return None
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidates = []
    if base_dir:
        candidates.append(os.path.join(base_dir, path))
    candidates.append(path)
    candidates.append(os.path.join(os.getcwd(), path))
    for c in candidates:
        if os.path.exists(c):
            return c
    return path


def list_lut_files(primary_path):
    """Return sorted, de-duplicated .cube files usable as LUT choices.

    Looks next to the configured LUT, in the current directory, and in LUT/,
    so files dropped in either conventional place show up in the TUI list.
    """
    folders = []
    if primary_path:
        folders.append(os.path.dirname(os.path.abspath(primary_path)))
    folders.append(os.getcwd())
    folders.append(os.path.join(os.getcwd(), "LUT"))
    found = {}
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for path in glob.glob(os.path.join(folder, "*.cube")):
            found.setdefault(os.path.abspath(path), None)
    return sorted(found)


def cycle_index(current, total, step):
    """Wrap a list index forward/backward, safe for an empty list."""
    if total <= 0:
        return 0
    return (current + step) % total


def build_lut_table(path):
    """Load and bake a .cube LUT into the runtime table; None when unusable."""
    lut = load_lut(path)
    return bake_lut(lut) if lut is not None else None
