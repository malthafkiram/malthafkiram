#!/usr/bin/env python3
"""Generate dark.svg / light.svg GitHub profile banners from source.jpg."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from scipy import ndimage
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
PROFILE_ROOT = ROOT.parent
SOURCE = ROOT / "source.jpg"
DARK_OUT = PROFILE_ROOT / "dark.svg"
LIGHT_OUT = PROFILE_ROOT / "light.svg"
PREVIEW_DIR = ROOT / "preview"

W, H = 1180, 610
GRID_W, GRID_H = 300, 340
INTRO_GROUPS = 60
DRIFT_BANDS = 94
TRAVELLERS = 900
NOISE_SIGMA = 4.0
INTRO_DUR = 3.2
LOOP_DUR = 14.2
KEY = [0.0, 3.0 / LOOP_DUR, 4.3 / LOOP_DUR, 6.3 / LOOP_DUR, 7.6 / LOOP_DUR, 9.6 / LOOP_DUR, 10.9 / LOOP_DUR, 12.9 / LOOP_DUR, 1.0]

PROFILE = {
    "name": "M. Althaf Kiram",
    "handle": "@malthafkiram",
    "subject": "M. Althaf Kiram",
    "role": "Fullstack Software Engineer",
    "origin": "Indonesia, remote or relocate",
    "education": "S.Kom Unimal, GPA 3.66",
    "status": "Available for hire",
    "toolchain": "VS Code, Git, Vercel",
    "lang": "TypeScript, JavaScript",
    "frontend": "React, Next.js, Tailwind",
    "backend": "Node.js, Express, GraphQL",
    "database": "PostgreSQL, MongoDB, Redis",
    "infra": "AWS EC2, Vercel",
    "mail": "malthafkiram@gmail.com",
    "portfolio": "porto.kabanroom.web.id",
    "linkedin": "/in/malthafkiram",
    "github": "github.com/malthafkiram",
    "whatsapp": "+62 851-5771-5522",
}

IDENTITY_ROWS = [
    ("Subject", "subject"),
    ("Role", "role"),
    ("Origin", "origin"),
    ("Education", "education"),
    ("Status", "status"),
    ("ToolChain", "toolchain"),
]
CORE_ROWS = [
    ("Core.Lang", "lang"),
    ("Core.Frontend", "frontend"),
    ("Core.Backend", "backend"),
    ("Core.Database", "database"),
    ("Core.Infra", "infra"),
]
GRID_ROWS = [
    ("Grid.Mail", "mail"),
    ("Grid.Portfolio", "portfolio"),
    ("Grid.LinkedIn", "linkedin"),
    ("Grid.GitHub", "github"),
    ("Grid.WhatsApp", "whatsapp"),
]


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    panel: str
    frame: str
    chrome: str
    chrome_dim: str
    portrait: str
    accent: str
    text: str
    text_dim: str
    live: str
    titlebar: str
    drop_background: bool


DARK = Theme(
    name="dark",
    bg="#0A101F",
    panel="#0E1628",
    frame="#152038",
    chrome="#22D3EE",
    chrome_dim="#0891B2",
    portrait="#A78BFA",
    accent="#10B981",
    text="#F8FAFC",
    text_dim="#94A3B8",
    live="#F43F5E",
    titlebar="#070B14",
    drop_background=True,
)
LIGHT = Theme(
    name="light",
    bg="#F4F7FB",
    panel="#E8EEF7",
    frame="#D5DEEC",
    chrome="#0891B2",
    chrome_dim="#0E7490",
    portrait="#7C3AED",
    accent="#059669",
    text="#0F172A",
    text_dim="#475569",
    live="#E11D48",
    titlebar="#E2E8F0",
    drop_background=False,
)


def xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def load_crop() -> Image.Image:
    img = Image.open(SOURCE).convert("RGB")
    w, h = img.size
    # Head and shoulders, not a tight face crop.
    left = int(w * 0.10)
    right = int(w * 0.90)
    top = int(h * 0.03)
    bottom = int(h * 0.58)
    return img.crop((left, top, right, bottom))


def subject_mask(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.where(mx == 0, 0, (mx - mn) / np.maximum(mx, 1e-6))
    wall = (sat < 0.18) & (mx > 0.72)
    fg = ~wall
    fg = ndimage.binary_opening(fg, iterations=2)
    fg = ndimage.binary_closing(fg, iterations=4)
    fg = ndimage.binary_fill_holes(fg)
    labeled, n = ndimage.label(fg)
    if n == 0:
        return np.ones(fg.shape, dtype=bool)
    sizes = ndimage.sum(fg, labeled, index=range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    fg = labeled == keep
    fg = ndimage.binary_erosion(fg, iterations=1)
    return fg


def enhance_gray(img: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.3)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140, threshold=2))
    return gray


def floyd_steinberg(gray: Image.Image, serpentine: bool = True) -> np.ndarray:
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    h, w = arr.shape
    out = np.zeros((h, w), dtype=bool)
    work = arr.copy()
    for y in range(h):
        xs = range(w)
        if serpentine and y % 2:
            xs = range(w - 1, -1, -1)
        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = new < 0.5  # dark pixels become dots
            err = old - new
            if serpentine and y % 2:
                if x - 1 >= 0:
                    work[y, x - 1] += err * 7 / 16
                if y + 1 < h:
                    if x + 1 < w:
                        work[y + 1, x + 1] += err * 3 / 16
                    work[y + 1, x] += err * 5 / 16
                    if x - 1 >= 0:
                        work[y + 1, x - 1] += err * 1 / 16
            else:
                if x + 1 < w:
                    work[y, x + 1] += err * 7 / 16
                if y + 1 < h:
                    if x - 1 >= 0:
                        work[y + 1, x - 1] += err * 3 / 16
                    work[y + 1, x] += err * 5 / 16
                    if x + 1 < w:
                        work[y + 1, x + 1] += err * 1 / 16
    return out


def dots_from_photo(theme: Theme) -> tuple[np.ndarray, Image.Image]:
    crop = load_crop()
    crop = crop.resize((GRID_W, GRID_H), Image.Resampling.LANCZOS)
    gray = enhance_gray(crop)
    dots = floyd_steinberg(gray)
    if theme.drop_background:
        mask = subject_mask(crop)
        mask = ndimage.binary_erosion(mask, iterations=1)
        dots &= mask
    return dots, crop


def thin_isolated(dots: np.ndarray) -> np.ndarray:
    neigh = ndimage.convolve(dots.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant")
    return dots & (neigh >= 3)


def sample_points(dots: np.ndarray, rng: np.random.Generator, cap: int = 17500) -> np.ndarray:
    dots = thin_isolated(dots)
    ys, xs = np.nonzero(dots)
    points = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    if len(points) > cap:
        idx = rng.choice(len(points), size=cap, replace=False)
        points = points[idx]
    return points


def evenness(points: np.ndarray, groups: np.ndarray, bins: int = 8) -> float:
    """Lower is better. ~0.05 even, ~0.7 patchy."""
    scores = []
    h = GRID_H
    w = GRID_W
    for g in range(int(groups.max()) + 1 if len(groups) else 0):
        pts = points[groups == g]
        if len(pts) < 4:
            continue
        hist, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins, range=[[0, w], [0, h]])
        p = hist.ravel()
        p = p / max(p.sum(), 1)
        scores.append(float(np.std(p)))
    return float(np.mean(scores)) if scores else 1.0


def interleave_groups(n: int, k: int, rng: random.Random) -> np.ndarray:
    idx = list(range(n))
    rng.shuffle(idx)
    groups = np.empty(n, dtype=np.int32)
    for i, src in enumerate(idx):
        groups[src] = i % k
    return groups


def drift_bands(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noisy = points + rng.normal(0, NOISE_SIGMA, size=points.shape)
    k = min(DRIFT_BANDS, max(8, len(points) // 80))
    for seed in range(7, 16):
        try:
            _, labels = kmeans2(noisy, k, minit="random", iter=30, seed=seed)
            if np.any(labels < 0):
                continue
            return labels.astype(np.int32)
        except Exception:
            continue
    order = np.argsort(noisy[:, 0] + noisy[:, 1] * 0.37)
    labels = np.empty(len(points), dtype=np.int32)
    labels[order] = np.linspace(0, k - 1, len(points)).astype(np.int32)
    return labels


def farthest_sample(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= n:
        return points.copy()
    chosen = [int(rng.integers(0, len(points)))]
    dist = np.full(len(points), np.inf)
    for _ in range(n - 1):
        last = points[chosen[-1]]
        dist = np.minimum(dist, np.linalg.norm(points - last, axis=1))
        dist[chosen] = -1
        chosen.append(int(np.argmax(dist)))
    return points[np.array(chosen)]


def match_ot(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    n = min(len(src), len(dst))
    src = src[:n]
    dst = dst[:n]
    diff = src[:, None, :] - dst[None, :, :]
    cost = np.sqrt((diff * diff).sum(axis=2))
    _, col = linear_sum_assignment(cost)
    return dst[col]


def logo_canvas(kind: str, size: int = 520) -> Image.Image:
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    m = int(size * 0.12)
    if kind == "next":
        d.ellipse((m, m, size - m, size - m), outline=255, width=max(10, size // 28))
        # Stylized N
        x0, x1 = int(size * 0.30), int(size * 0.70)
        y0, y1 = int(size * 0.28), int(size * 0.74)
        wline = max(16, size // 22)
        d.line([(x0, y1), (x0, y0), (x1, y1), (x1, y0)], fill=255, width=wline, joint="curve")
        d.line([(x0, y0), (x1, y1)], fill=255, width=wline)
    elif kind == "node":
        cx, cy, r = size / 2, size / 2, size * 0.36
        hexpts = [
            (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
            for a in range(30, 360, 60)
        ]
        d.polygon(hexpts, outline=255, width=max(12, size // 24))
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=int(size * 0.28)
            )
        except OSError:
            font = ImageFont.load_default()
        d.text((cx, cy + size * 0.02), "JS", fill=255, font=font, anchor="mm")
    else:
        wline = max(18, size // 18)
        # < / >
        d.line(
            [(size * 0.46, size * 0.22), (size * 0.22, size * 0.50), (size * 0.46, size * 0.78)],
            fill=255,
            width=wline,
            joint="curve",
        )
        d.line(
            [(size * 0.54, size * 0.22), (size * 0.78, size * 0.50), (size * 0.54, size * 0.78)],
            fill=255,
            width=wline,
            joint="curve",
        )
        d.line(
            [(size * 0.58, size * 0.24), (size * 0.42, size * 0.76)],
            fill=255,
            width=max(14, size // 22),
        )
    return img


def logo_points(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    canvas = logo_canvas(kind)
    arr = np.asarray(canvas)
    ys, xs = np.nonzero(arr > 80)
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    # Fit into portrait grid with margin
    pts[:, 0] = (pts[:, 0] / canvas.size[0]) * (GRID_W * 0.78) + GRID_W * 0.11
    pts[:, 1] = (pts[:, 1] / canvas.size[1]) * (GRID_H * 0.70) + GRID_H * 0.14
    return farthest_sample(pts, n, rng)


def path_for_points(points: np.ndarray, ox: float, oy: float, scale: float, dot: float) -> str:
    # Horizontal run-length packing in grid space, then map to SVG.
    if len(points) == 0:
        return ""
    by_row: dict[int, list[int]] = {}
    for x, y in points:
        by_row.setdefault(int(round(y)), []).append(int(round(x)))
    parts: list[str] = []
    half = dot * 0.42
    for y in sorted(by_row):
        xs = sorted(set(by_row[y]))
        run_start = xs[0]
        prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            length = prev - run_start + 1
            sx = ox + run_start * scale
            sy = oy + y * scale
            if length == 1:
                parts.append(f"M{sx:.2f} {sy:.2f}h{dot:.2f}")
            else:
                parts.append(f"M{sx:.2f} {sy:.2f}h{length * scale + half:.2f}")
            if x is None:
                break
            run_start = prev = x
    return " ".join(parts)


def key_times(*indices: int) -> str:
    return ";".join(f"{KEY[i]:.4f}" for i in indices)


def portrait_frame_geom() -> tuple[float, float, float, float]:
    # Left visual panel inner rect
    panel_x, panel_y = 28, 78
    panel_w, panel_h = 430, 506
    scale = min((panel_w - 36) / GRID_W, (panel_h - 56) / GRID_H)
    pw, ph = GRID_W * scale, GRID_H * scale
    ox = panel_x + (panel_w - pw) / 2
    oy = panel_y + 28 + (panel_h - 40 - ph) / 2
    return ox, oy, scale, scale * 0.86


def leader_line(label: str, value: str, width: int = 52) -> str:
    dots = max(3, width - len(label) - len(value))
    return f"{label}{'.' * dots}{value}"


def info_panel_svg(theme: Theme) -> str:
    x, y = 478, 78
    width, height = 674, 506
    rows_y = y + 64
    line_h = 23
    chunks = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="{theme.panel}" stroke="{theme.frame}" stroke-width="1.2"/>',
        f'<text x="{x + 22}" y="{y + 32}" fill="{theme.chrome}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="13" font-weight="700" letter-spacing="1.6">SYSTEM.INFO</text>',
        f'<g transform="translate({x + 168},{y + 20})">',
        f'<rect width="64" height="18" rx="9" fill="{theme.live}"/>',
        f'<circle cx="12" cy="9" r="3.2" fill="#fff"><animate attributeName="opacity" values="1;0.25;1" dur="1.5s" repeatCount="indefinite"/></circle>',
        f'<text x="22" y="13" fill="#fff" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="12" font-weight="700">LIVE</text>',
        "</g>",
        f'<rect x="{x + width - 196}" y="{y + 16}" width="174" height="26" rx="13" fill="{theme.bg}" stroke="{theme.chrome}" stroke-width="1"/>',
        f'<text x="{x + width - 109}" y="{y + 34}" text-anchor="middle" fill="{theme.chrome}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="14">{xml_escape(PROFILE["handle"])}</text>',
    ]

    def emit_section(title: str, rows: list[tuple[str, str]], start: float) -> float:
        chunks.append(
            f'<text x="{x + 22}" y="{start}" fill="{theme.accent}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="12" letter-spacing="1.4">{title}</text>'
        )
        yy = start + 22
        text_w = width - 44
        for label, key in rows:
            line = xml_escape(leader_line(label, PROFILE[key]))
            chunks.append(
                f'<text x="{x + 22}" y="{yy}" fill="{theme.text}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="14" textLength="{text_w}" lengthAdjust="spacingAndGlyphs">{line}</text>'
            )
            yy += line_h
        return yy + 10

    cursor = rows_y
    cursor = emit_section("// IDENTITY", IDENTITY_ROWS, cursor)
    cursor = emit_section("// CORE", CORE_ROWS, cursor)
    emit_section("// GRID", GRID_ROWS, cursor)
    return "\n".join(chunks)


def chrome_svg(theme: Theme) -> str:
    return f"""
<rect width="{W}" height="{H}" rx="28" fill="{theme.bg}"/>
<rect x="10" y="10" width="{W-20}" height="{H-20}" rx="22" fill="{theme.bg}" stroke="{theme.chrome}" stroke-opacity="0.35" stroke-width="1.4"/>
<rect x="10" y="10" width="{W-20}" height="52" rx="22" fill="{theme.titlebar}"/>
<rect x="10" y="36" width="{W-20}" height="26" fill="{theme.titlebar}"/>
<circle cx="38" cy="36" r="6" fill="#F43F5E"/>
<circle cx="58" cy="36" r="6" fill="#FBBF24"/>
<circle cx="78" cy="36" r="6" fill="{theme.accent}"/>
<text x="{W/2}" y="41" text-anchor="middle" fill="{theme.text_dim}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="14">profile.sh --live</text>
<rect x="28" y="78" width="430" height="506" rx="18" fill="{theme.panel}" stroke="{theme.frame}" stroke-width="1.2"/>
<text x="50" y="106" fill="{theme.chrome}" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="13" font-weight="700" letter-spacing="1.6">VISUAL.MAP</text>
"""


def group_path(points: np.ndarray, ox: float, oy: float, scale: float, dot: float, color: str) -> str:
    d = path_for_points(points, ox, oy, scale, dot)
    if not d:
        return ""
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{dot:.2f}" stroke-linecap="square" shape-rendering="crispEdges"/>'


def intro_layer(points: np.ndarray, groups: np.ndarray, theme: Theme, ox: float, oy: float, scale: float, dot: float) -> str:
    parts = ['<g id="intro">']
    for g in range(INTRO_GROUPS):
        pts = points[groups == g]
        if len(pts) == 0:
            continue
        delay = (g / INTRO_GROUPS) * 2.0
        t1 = delay / INTRO_DUR
        t2 = min(0.94, (delay + 0.45) / INTRO_DUR)
        path = group_path(pts, ox, oy, scale, dot, theme.portrait)
        parts.append(
            f'<g opacity="1">{path}'
            f'<animate attributeName="opacity" values="0;0;1;1;0" keyTimes="0;{t1:.3f};{t2:.3f};0.96;1" '
            f'dur="{INTRO_DUR}s" begin="0s" fill="freeze"/>'
            "</g>"
        )
    parts.append("</g>")
    return "\n".join(parts)


def loop_layer(
    points: np.ndarray,
    bands: np.ndarray,
    logo_centroid: np.ndarray,
    theme: Theme,
    ox: float,
    oy: float,
    scale: float,
    dot: float,
) -> str:
    parts = [
        '<g id="loop-portrait" opacity="0">',
        f'<animate attributeName="opacity" values="0;1" dur="0.05s" begin="{INTRO_DUR}s" fill="freeze"/>',
    ]
    kt = f"{KEY[0]:.4f};{KEY[1]:.4f};{KEY[2]:.4f};{KEY[7]:.4f};{KEY[8]:.4f}"
    for b in range(int(bands.max()) + 1 if len(bands) else 0):
        pts = points[bands == b]
        if len(pts) == 0:
            continue
        c = pts.mean(axis=0)
        target = c + 0.42 * (logo_centroid - c)
        dx = (target[0] - c[0]) * scale
        dy = (target[1] - c[1]) * scale
        path = group_path(pts, ox, oy, scale, dot, theme.portrait)
        parts.append(
            f'<g transform="translate(0,0)">{path}'
            f'<animateTransform attributeName="transform" type="translate" '
            f'values="0,0; 0,0; {dx:.2f},{dy:.2f}; {dx:.2f},{dy:.2f}; 0,0" '
            f'keyTimes="{kt}" dur="{LOOP_DUR}s" begin="{INTRO_DUR}s" repeatCount="indefinite" calcMode="linear"/>'
            f'<animate attributeName="opacity" values="1;1;0;0;1" keyTimes="{kt}" '
            f'dur="{LOOP_DUR}s" begin="{INTRO_DUR}s" repeatCount="indefinite" calcMode="linear"/>'
            "</g>"
        )
    parts.append("</g>")
    return "\n".join(parts)


def travellers_layer(
    portrait: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    l3: np.ndarray,
    theme: Theme,
    ox: float,
    oy: float,
    scale: float,
    dot: float,
) -> str:
    kt_pos = ";".join(f"{KEY[i]:.4f}" for i in range(9))
    kt_op = f"{KEY[0]:.4f};{KEY[1]:.4f};{KEY[2]:.4f};{KEY[7]:.4f};{KEY[8]:.4f}"
    tdot = min(2.15, dot * 1.55)
    parts = ['<g id="travellers">']
    for i in range(len(portrait)):
        chain = [portrait[i], portrait[i], l1[i], l1[i], l2[i], l2[i], l3[i], l3[i], portrait[i]]
        xs = ";".join(f"{ox + p[0] * scale:.2f}" for p in chain)
        ys = ";".join(f"{oy + p[1] * scale:.2f}" for p in chain)
        x0, y0 = ox + portrait[i][0] * scale, oy + portrait[i][1] * scale
        parts.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{tdot:.2f}" height="{tdot:.2f}" rx="0.4" fill="{theme.chrome}" opacity="0" shape-rendering="crispEdges">'
            f'<animate attributeName="x" values="{xs}" keyTimes="{kt_pos}" dur="{LOOP_DUR}s" begin="{INTRO_DUR}s" repeatCount="indefinite" calcMode="linear"/>'
            f'<animate attributeName="y" values="{ys}" keyTimes="{kt_pos}" dur="{LOOP_DUR}s" begin="{INTRO_DUR}s" repeatCount="indefinite" calcMode="linear"/>'
            f'<animate attributeName="opacity" values="0;0;1;1;0" keyTimes="{kt_op}" dur="{LOOP_DUR}s" begin="{INTRO_DUR}s" repeatCount="indefinite" calcMode="linear"/>'
            "</rect>"
        )
    parts.append("</g>")
    return "\n".join(parts)


def build_svg(theme: Theme) -> str:
    rng = random.Random(42)
    nprng = np.random.default_rng(42)
    dots, _crop = dots_from_photo(theme)
    np.save(PREVIEW_DIR / f"{theme.name}-dots.npy", dots)
    points = sample_points(dots, nprng)
    print(f"[{theme.name}] dots={len(points)}")
    intro_groups = interleave_groups(len(points), INTRO_GROUPS, rng)
    print(f"[{theme.name}] intro evenness={evenness(points, intro_groups):.3f}")
    bands = drift_bands(points, nprng)
    travellers_src = farthest_sample(points, TRAVELLERS, nprng)
    n = len(travellers_src)
    l1 = match_ot(travellers_src, logo_points("next", n, nprng))
    l2 = match_ot(l1, logo_points("node", n, nprng))
    l3 = match_ot(l2, logo_points("code", n, nprng))
    ox, oy, scale, dot = portrait_frame_geom()
    logo_centroid = l1.mean(axis=0)
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{xml_escape(PROFILE["name"])} - live profile">
<style>
  @media (prefers-reduced-motion: reduce) {{
    animate, animateTransform {{ display: none; }}
  }}
</style>
{chrome_svg(theme)}
{info_panel_svg(theme)}
{intro_layer(points, intro_groups, theme, ox, oy, scale, dot)}
{loop_layer(points, bands, logo_centroid, theme, ox, oy, scale, dot)}
{travellers_layer(travellers_src, l1, l2, l3, theme, ox, oy, scale, dot)}
</svg>
'''
    return svg


def preview_png(theme: Theme, path: Path) -> None:
    dots, crop = dots_from_photo(theme)
    img = Image.new("RGB", (W, H), theme.bg)
    draw = ImageDraw.Draw(img)
    ox, oy, scale, dot = portrait_frame_geom()
    # terminal-ish background
    draw.rounded_rectangle((10, 10, W - 10, H - 10), radius=22, outline=theme.chrome, width=2)
    ys, xs = np.nonzero(dots)
    color = tuple(int(theme.portrait[i : i + 2], 16) for i in (1, 3, 5))
    r = max(1, int(dot))
    for x, y in zip(xs, ys):
        px, py = ox + x * scale, oy + y * scale
        draw.rectangle((px, py, px + r, py + r), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    crop.save(path.with_name(f"{theme.name}-crop.jpg"), quality=92)


def main() -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for theme, out in ((DARK, DARK_OUT), (LIGHT, LIGHT_OUT)):
        preview_png(theme, PREVIEW_DIR / f"{theme.name}-frame.png")
        svg = build_svg(theme)
        out.write_text(svg, encoding="utf-8")
        kb = out.stat().st_size / 1024
        print(f"wrote {out.name} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
