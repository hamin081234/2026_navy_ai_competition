from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile
from scipy.signal import stft


BASE = Path("database")
OUT = Path("task2_eda_visualizations")
FONT = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"

PALETTE = {
    "A_SmallWorking": (44, 123, 182),
    "B_MotorBoat": (230, 120, 45),
    "C_Passenger": (74, 150, 98),
    "D_LargeShip": (150, 85, 170),
    "train": (84, 112, 198),
    "gallery": (40, 145, 135),
    "val": (220, 132, 45),
    "grid": (224, 228, 234),
    "axis": (70, 76, 86),
    "text": (28, 32, 38),
    "muted": (96, 104, 116),
    "bg": (255, 255, 255),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def canvas(w: int = 1400, h: int = 900) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w, h), PALETTE["bg"])
    return img, ImageDraw.Draw(img)


def text(draw: ImageDraw.ImageDraw, xy, s: str, size=22, fill=None, bold=False):
    draw.text(xy, s, font=font(size, bold), fill=fill or PALETTE["text"])


def title(draw, main: str, sub: str | None = None):
    text(draw, (42, 28), main, 34, bold=True)
    if sub:
        text(draw, (44, 76), sub, 19, fill=PALETTE["muted"])


def draw_axes(draw, box, x_label="", y_label="", y_ticks=None):
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=PALETTE["axis"], width=2)
    draw.line((x0, y0, x0, y1), fill=PALETTE["axis"], width=2)
    if y_ticks:
        for val, lab in y_ticks:
            y = y1 - val * (y1 - y0)
            draw.line((x0 - 6, y, x0, y), fill=PALETTE["axis"], width=2)
            draw.line((x0, y, x1, y), fill=PALETTE["grid"], width=1)
            text(draw, (x0 - 76, y - 11), lab, 15, fill=PALETTE["muted"])
    if x_label:
        text(draw, ((x0 + x1) // 2 - 60, y1 + 50), x_label, 18, fill=PALETTE["muted"])
    if y_label:
        text(draw, (x0 - 78, y0 - 34), y_label, 18, fill=PALETTE["muted"])


def save(img: Image.Image, name: str):
    OUT.mkdir(exist_ok=True)
    img.save(OUT / name, optimize=True)
    print(OUT / name)


def bar_chart_counts(data: dict[str, dict[str, int]], name: str):
    img, draw = canvas()
    title(draw, "Class distribution by split", "Task 2 is ship-id retrieval, but class imbalance still affects embeddings.")
    classes = ["A_SmallWorking", "B_MotorBoat", "C_Passenger", "D_LargeShip"]
    splits = list(data)
    box = (150, 170, 1320, 720)
    maxv = max(max(d.values()) for d in data.values())
    maxv = math.ceil(maxv / 5000) * 5000
    ticks = [(i / 4, f"{int(maxv * i / 4):,}") for i in range(5)]
    draw_axes(draw, box, y_label="clips", y_ticks=ticks)
    group_w = (box[2] - box[0]) / len(splits)
    bar_w = 46
    for i, split in enumerate(splits):
        cx = box[0] + group_w * (i + 0.5)
        offsets = [-1.5, -0.5, 0.5, 1.5]
        for off, cls in zip(offsets, classes):
            v = data[split].get(cls, 0)
            h = (v / maxv) * (box[3] - box[1])
            x = cx + off * (bar_w + 8)
            draw.rectangle((x - bar_w / 2, box[3] - h, x + bar_w / 2, box[3]), fill=PALETTE[cls])
            text(draw, (x - 22, box[3] - h - 26), f"{v:,}", 12, fill=PALETTE["muted"])
        text(draw, (cx - 44, box[3] + 16), split, 18, bold=True)
    lx, ly = 190, 785
    for cls in classes:
        draw.rectangle((lx, ly, lx + 24, ly + 24), fill=PALETTE[cls])
        text(draw, (lx + 34, ly - 1), cls, 17)
        lx += 285
    save(img, name)


def ship_count_distribution(train, gallery, val):
    img, draw = canvas(1500, 900)
    title(draw, "Clips per ship", "Gallery imbalance matters: clip-level nearest neighbor can favor ships with many gallery clips.")
    series = {
        "train": train.ship_id.value_counts().sort_values().values,
        "gallery": gallery.ship_id.value_counts().sort_values().values,
        "val": val.ship_id.value_counts().sort_values().values,
    }
    boxes = {
        "train": (120, 170, 1440, 330),
        "gallery": (120, 410, 1440, 570),
        "val": (120, 650, 1440, 810),
    }
    maxv = max(v.max() for v in series.values())
    for split, vals in series.items():
        box = boxes[split]
        draw_axes(draw, box, x_label="ships sorted by clip count", y_label=split)
        n = len(vals)
        for i, v in enumerate(vals):
            x0 = box[0] + int(i * (box[2] - box[0]) / n)
            x1 = box[0] + int((i + 1) * (box[2] - box[0]) / n)
            y = box[3] - int(v / maxv * (box[3] - box[1]))
            draw.rectangle((x0, y, max(x1, x0 + 1), box[3]), fill=PALETTE[split])
        text(draw, (box[2] - 270, box[1] + 8), f"min={vals.min()} median={np.median(vals):.1f} max={vals.max()}", 18, bold=True)
    save(img, "02_ship_clip_distribution.png")


def gallery_val_scatter(gallery, val):
    img, draw = canvas(1300, 900)
    title(draw, "Gallery vs validation clips per ship", "Outliers show where retrieval evaluation may be unstable or gallery coverage is thin.")
    g = gallery.ship_id.value_counts()
    v = val.ship_id.value_counts()
    ships = sorted(set(g.index) | set(v.index))
    xs = np.array([g.get(s, 0) for s in ships], dtype=float)
    ys = np.array([v.get(s, 0) for s in ships], dtype=float)
    box = (150, 150, 1180, 760)
    xmax = math.ceil(xs.max() / 50) * 50
    ymax = math.ceil(ys.max() / 20) * 20
    draw_axes(draw, box, "gallery clips", "val clips", [(i / 5, f"{int(ymax*i/5)}") for i in range(6)])
    for i in range(6):
        x = box[0] + i / 5 * (box[2] - box[0])
        lab = f"{int(xmax*i/5)}"
        draw.line((x, box[3], x, box[3] + 6), fill=PALETTE["axis"], width=2)
        text(draw, (x - 16, box[3] + 14), lab, 15, fill=PALETTE["muted"])
    for sid, x, y in zip(ships, xs, ys):
        px = box[0] + x / xmax * (box[2] - box[0])
        py = box[3] - y / ymax * (box[3] - box[1])
        cls = gallery.loc[gallery.ship_id == sid, "ship_type"].iloc[0]
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=PALETTE[cls])
    label_ids = [94, 136, 242, 30, 276, 155]
    for sid in label_ids:
        if sid not in ships:
            continue
        x, y = g.get(sid, 0), v.get(sid, 0)
        px = box[0] + x / xmax * (box[2] - box[0])
        py = box[3] - y / ymax * (box[3] - box[1])
        text(draw, (px + 8, py - 12), str(sid), 16, bold=True)
    save(img, "03_gallery_val_ship_scatter.png")


def ais_distributions(train, gallery):
    img, draw = canvas(1500, 980)
    title(draw, "AIS distributions", "AIS is unavailable for Task 2 query/test, so treat it as analysis or auxiliary training signal.")
    panels = [
        ("train SOG", train.sog.clip(0, 30), (90, 160, 700, 430), PALETTE["train"], "knots"),
        ("gallery SOG", gallery.sog.clip(0, 30), (820, 160, 1430, 430), PALETTE["gallery"], "knots"),
        ("train true_heading", train.true_heading.clip(0, 360), (90, 590, 700, 860), PALETTE["train"], "degrees"),
        ("gallery true_heading", gallery.true_heading.clip(0, 360), (820, 590, 1430, 860), PALETTE["gallery"], "degrees"),
    ]
    for name, values, box, color, xlabel in panels:
        vals = np.asarray(values.dropna())
        bins = 30
        hist, edges = np.histogram(vals, bins=bins)
        maxv = hist.max()
        draw_axes(draw, box, xlabel, "count", [(i / 4, f"{int(maxv*i/4):,}") for i in range(5)])
        for i, h in enumerate(hist):
            x0 = box[0] + i * (box[2] - box[0]) / bins
            x1 = box[0] + (i + 1) * (box[2] - box[0]) / bins
            y = box[3] - h / maxv * (box[3] - box[1])
            draw.rectangle((x0 + 1, y, x1 - 1, box[3]), fill=color)
        zero_ratio = float((values == 0).mean())
        text(draw, (box[0] + 8, box[1] - 36), f"{name} | zero={zero_ratio:.1%}", 20, bold=True)
    save(img, "04_ais_distributions.png")


def month_counts(train, gallery):
    img, draw = canvas(1400, 820)
    title(draw, "Monthly timestamp distribution", "Train and gallery are not temporally identical; time effects can leak into learned embeddings.")
    t = pd.to_datetime(train.ais_timestamp, format="mixed", utc=True).dt.to_period("M").astype(str).value_counts().sort_index()
    g = pd.to_datetime(gallery.ais_timestamp, format="mixed", utc=True).dt.to_period("M").astype(str).value_counts().sort_index()
    months = sorted(set(t.index) | set(g.index))
    box = (130, 160, 1300, 650)
    maxv = max(t.max(), g.max())
    draw_axes(draw, box, "month", "clips", [(i / 5, f"{int(maxv*i/5):,}") for i in range(6)])
    group_w = (box[2] - box[0]) / len(months)
    for i, m in enumerate(months):
        cx = box[0] + group_w * (i + 0.5)
        for off, series, color in [(-18, t, PALETTE["train"]), (18, g, PALETTE["gallery"])]:
            v = series.get(m, 0)
            h = v / maxv * (box[3] - box[1])
            draw.rectangle((cx + off - 14, box[3] - h, cx + off + 14, box[3]), fill=color)
        text(draw, (cx - 34, box[3] + 18), m, 14, fill=PALETTE["muted"])
    draw.rectangle((170, 710, 194, 734), fill=PALETTE["train"])
    text(draw, (204, 707), "train", 18)
    draw.rectangle((300, 710, 324, 734), fill=PALETTE["gallery"])
    text(draw, (334, 707), "gallery", 18)
    save(img, "05_monthly_timestamp_distribution.png")


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(path)
    x = x.astype(np.float32)
    peak = np.max(np.abs(x)) + 1e-9
    return sr, x / peak


def log_spectrogram(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    freqs, _, z = stft(x, fs=sr, nperseg=1024, noverlap=768, boundary=None, padded=False)
    spec = np.log10(np.abs(z) ** 2 + 1e-10)
    keep = freqs <= 8000
    return freqs[keep], spec[keep]


def heat_color(v: float) -> tuple[int, int, int]:
    # Dark blue -> teal -> yellow, chosen for spectral contrast.
    stops = [
        (0.0, (13, 22, 46)),
        (0.35, (35, 86, 125)),
        (0.68, (35, 151, 139)),
        (1.0, (245, 204, 85)),
    ]
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if a <= v <= b:
            t = (v - a) / (b - a)
            return tuple(int(ca[i] * (1 - t) + cb[i] * t) for i in range(3))
    return stops[-1][1]


def spec_image(spec: np.ndarray, size: tuple[int, int]) -> Image.Image:
    lo, hi = np.percentile(spec, [5, 99])
    arr = np.clip((spec - lo) / (hi - lo + 1e-9), 0, 1)
    arr = arr[::-1]
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    flat = arr.ravel()
    colors = np.array([heat_color(float(v)) for v in flat], dtype=np.uint8)
    rgb.reshape(-1, 3)[:] = colors
    return Image.fromarray(rgb, "RGB").resize(size, Image.Resampling.BILINEAR)


def waveform_panel(draw, x: np.ndarray, box, color=(46, 90, 160)):
    x0, y0, x1, y1 = box
    mid = (y0 + y1) / 2
    amp = (y1 - y0) * 0.45
    n = x1 - x0
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    vals = x[idx]
    pts = [(x0 + i, mid - float(v) * amp) for i, v in enumerate(vals)]
    draw.rectangle(box, outline=(222, 226, 232))
    if len(pts) > 1:
        draw.line(pts, fill=color, width=1)
    draw.line((x0, mid, x1, mid), fill=(220, 224, 230), width=1)


def choose_representatives(gallery):
    rows = []
    for cls in ["A_SmallWorking", "B_MotorBoat", "C_Passenger", "D_LargeShip"]:
        sub = gallery[gallery.ship_type == cls].copy()
        moving = sub[sub.sog > 0]
        if len(moving):
            sub = moving
        med = sub.sog.median()
        idx = (sub.sog - med).abs().sort_values().index[0]
        rows.append(gallery.loc[idx])
    return rows


def class_audio_overview(gallery):
    reps = choose_representatives(gallery)
    img, draw = canvas(1600, 1280)
    title(draw, "Representative WAV views by class", "Each row shows waveform and log spectrogram up to 8 kHz for one gallery clip.")
    y = 140
    for row in reps:
        p = BASE / "task2_test/audio" / row.filename
        sr, x = read_audio(p)
        freqs, spec = log_spectrogram(x, sr)
        color = PALETTE[row.ship_type]
        text(draw, (56, y), f"{row.ship_type} | ship {int(row.ship_id)} | {row.filename} | SOG {row.sog:.1f}", 22, bold=True, fill=color)
        waveform_panel(draw, x, (60, y + 40, 480, y + 245), color)
        simg = spec_image(spec, (960, 205))
        img.paste(simg, (560, y + 40))
        draw.rectangle((560, y + 40, 1520, y + 245), outline=(222, 226, 232))
        text(draw, (60, y + 254), "waveform", 15, fill=PALETTE["muted"])
        text(draw, (560, y + 254), "log spectrogram: time ->, frequency 0-8kHz bottom-to-top", 15, fill=PALETTE["muted"])
        y += 285
    save(img, "06_audio_overview_by_class.png")


def ship_variability(gallery, ship_id=30):
    sub = gallery[gallery.ship_id == ship_id].sort_values("ais_timestamp")
    if len(sub) < 6:
        sub = gallery[gallery.ship_id == gallery.ship_id.value_counts().idxmax()].sort_values("ais_timestamp")
        ship_id = int(sub.ship_id.iloc[0])
    idxs = np.linspace(0, len(sub) - 1, 6).astype(int)
    rows = sub.iloc[idxs]
    img, draw = canvas(1600, 1050)
    title(draw, f"Within-ship variation: ship {ship_id}", "Same ship can look different across clips; retrieval should learn stable cues, not one static spectrum.")
    x_positions = [60, 560, 1060]
    y_positions = [150, 560]
    for row, x0, y0 in zip(rows.itertuples(), x_positions * 2, [y_positions[0]] * 3 + [y_positions[1]] * 3):
        sr, x = read_audio(BASE / "task2_test/audio" / row.filename)
        _, spec = log_spectrogram(x, sr)
        simg = spec_image(spec, (420, 250))
        img.paste(simg, (x0, y0 + 42))
        draw.rectangle((x0, y0 + 42, x0 + 420, y0 + 292), outline=(222, 226, 232))
        text(draw, (x0, y0), f"{row.filename}", 18, bold=True)
        text(draw, (x0, y0 + 302), f"SOG {row.sog:.1f} | COG {row.cog:.1f} | heading {row.true_heading}", 16, fill=PALETTE["muted"])
    save(img, "07_same_ship_variability.png")


def scarce_gallery_case(gallery, val, ship_id=94):
    if ship_id not in set(gallery.ship_id):
        ship_id = int(gallery.ship_id.value_counts().sort_values().index[0])
    g = gallery[gallery.ship_id == ship_id].head(1)
    v = val[val.ship_id == ship_id].head(5)
    rows = [("gallery", r) for r in g.itertuples()] + [("val", r) for r in v.itertuples()]
    img, draw = canvas(1600, 1220)
    title(draw, f"Low-gallery ship case: ship {ship_id}", "A single gallery reference must cover many query variants, so augmentation and robust pooling matter.")
    for i, (split, row) in enumerate(rows):
        col = i % 2
        r = i // 2
        x0 = 70 + col * 760
        y0 = 145 + r * 335
        sr, x = read_audio(BASE / "task2_test/audio" / row.filename)
        _, spec = log_spectrogram(x, sr)
        text(draw, (x0, y0), f"{split}: {row.filename}", 20, bold=True, fill=PALETTE["gallery"] if split == "gallery" else PALETTE["val"])
        waveform_panel(draw, x, (x0, y0 + 38, x0 + 660, y0 + 122), (52, 88, 150))
        img.paste(spec_image(spec, (660, 170)), (x0, y0 + 136))
        draw.rectangle((x0, y0 + 136, x0 + 660, y0 + 306), outline=(222, 226, 232))
    save(img, "08_low_gallery_ship_case.png")


def audio_feature_scatter(gallery):
    # Deterministic stratified sample to keep runtime practical.
    samples = []
    for cls in ["A_SmallWorking", "B_MotorBoat", "C_Passenger", "D_LargeShip"]:
        sub = gallery[gallery.ship_type == cls].sort_values("filename")
        n = min(280, len(sub))
        idx = np.linspace(0, len(sub) - 1, n).astype(int)
        samples.append(sub.iloc[idx])
    sample = pd.concat(samples)
    rows = []
    for row in sample.itertuples():
        sr, x = read_audio(BASE / "task2_test/audio" / row.filename)
        freqs, spec = log_spectrogram(x, sr)
        power = np.maximum(10 ** spec, 1e-12)
        mean_power = power.mean(axis=1)
        centroid = float((freqs * mean_power).sum() / mean_power.sum())
        rms = float(np.sqrt(np.mean(x * x)))
        rows.append((rms, centroid, row.ship_type))
    img, draw = canvas(1300, 900)
    title(draw, "Audio feature scatter", "Simple RMS and spectral centroid separate broad conditions, but not ship identity by themselves.")
    box = (150, 150, 1180, 740)
    xs = np.array([r[0] for r in rows])
    ys = np.array([r[1] for r in rows])
    xmax = np.percentile(xs, 99) * 1.05
    ymax = 8000
    draw_axes(draw, box, "RMS loudness", "spectral centroid Hz", [(i / 4, f"{int(ymax*i/4)}") for i in range(5)])
    for rms, centroid, cls in rows:
        px = box[0] + min(rms, xmax) / xmax * (box[2] - box[0])
        py = box[3] - min(centroid, ymax) / ymax * (box[3] - box[1])
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=PALETTE[cls])
    lx, ly = 170, 790
    for cls in ["A_SmallWorking", "B_MotorBoat", "C_Passenger", "D_LargeShip"]:
        draw.ellipse((lx, ly, lx + 18, ly + 18), fill=PALETTE[cls])
        text(draw, (lx + 28, ly - 4), cls, 17)
        lx += 275
    save(img, "09_audio_feature_scatter.png")


def main():
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(BASE / "train/train.csv")
    gallery = pd.read_csv(BASE / "task2_test/gallery.csv")
    val = pd.read_csv(BASE / "task2_test/val.csv")

    bar_chart_counts(
        {
            "train": train.ship_type.value_counts().to_dict(),
            "gallery": gallery.ship_type.value_counts().to_dict(),
            "val": val.ship_type.value_counts().to_dict(),
        },
        "01_class_distribution.png",
    )
    ship_count_distribution(train, gallery, val)
    gallery_val_scatter(gallery, val)
    ais_distributions(train, gallery)
    month_counts(train, gallery)
    class_audio_overview(gallery)
    ship_variability(gallery, ship_id=30)
    scarce_gallery_case(gallery, val, ship_id=94)
    audio_feature_scatter(gallery)


if __name__ == "__main__":
    main()
