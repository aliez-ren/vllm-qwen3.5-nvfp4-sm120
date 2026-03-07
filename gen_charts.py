#!/usr/bin/env python3
"""Generate SVG benchmark charts from README data."""

import math

# Data: depth -> value for each config
DEPTHS = [2048, 4096, 8192, 131072]
DEPTH_LABELS = ["2K", "4K", "8K", "128K"]

# === Concurrency=1, Marlin ===
c1_marlin = {
    "pp_ts":      [4061.39, 4016.16, 3942.72, 2495.71],
    "tg_ts":      [80.12, 79.69, 79.22, 70.04],
    "tg_peak":    [82.76, 82.31, 81.83, 72.49],
    "ttfr":       [1012.92, 1534.21, 2601.76, 53344.75],
    "e2e_ttft":   [1013.21, 1534.34, 2601.91, 53344.93],
}

# === Concurrency=1, FlashInfer-CUTLASS ===
c1_flash = {
    "pp_ts":      [10276.71, 13688.11, 12640.29, 4552.98],
    "tg_ts":      [58.54, 58.23, 58.19, 53.39],
    "tg_peak":    [60.46, 60.14, 60.09, 55.19],
    "ttfr":       [460.19, 453.48, 814.68, 29247.07],
    "e2e_ttft":   [460.41, 453.63, 814.86, 29247.22],
}

# === Concurrency=4, Marlin ===
c4_marlin = {
    "pp_ts":      [4046.26, 3979.38, 3876.30, 2477.61],
    "pp_ts_req":  [1394.19, 1412.82, 1552.86, 1284.53],
    "tg_ts":      [47.73, 32.72, 17.57, 0.72],
    "tg_ts_req":  [35.14, 30.29, 23.94, 69.66],
    "tg_peak":    [124.00, 121.00, 116.00, 30.00],
    "tg_peak_req":[41.90, 39.99, 38.19, 72.11],
    "ttfr":       [3196.22, 4759.99, 7593.40, 134608.95],
    "e2e_ttft":   [3196.32, 4760.10, 7593.49, 134609.12],
}

# === Concurrency=4, FlashInfer-CUTLASS ===
c4_flash = {
    "pp_ts":      [14471.27, 14276.57, 13229.26, 4445.64],
    "pp_ts_req":  [5104.29, 5154.74, 5414.06, 2326.61],
    "tg_ts":      [104.41, 80.77, 48.66, 1.32],
    "tg_ts_req":  [41.84, 37.34, 28.59, 53.51],
    "tg_peak":    [126.33, 124.00, 116.00, 31.00],
    "tg_peak_req":[43.71, 40.34, 34.99, 55.32],
    "ttfr":       [884.55, 1318.28, 2188.02, 74729.35],
    "e2e_ttft":   [884.66, 1318.36, 2188.14, 74729.50],
}

COLORS = {
    "C1 Marlin":    "#2563eb",  # blue  = Marlin
    "C1 FlashInfer":"#dc2626",  # red   = FlashInfer
    "C4 Marlin":    "#2563eb",  # blue  = Marlin
    "C4 FlashInfer":"#dc2626",  # red   = FlashInfer
}

DASH = {
    "C1 Marlin":    "",         # solid = C1
    "C1 FlashInfer":"",         # solid = C1
    "C4 Marlin":    "8,4",      # dashed = C4
    "C4 FlashInfer":"8,4",      # dashed = C4
}

# Chart dimensions
W, H = 720, 420
PAD_L, PAD_R, PAD_T, PAD_B = 90, 30, 50, 70
PLOT_W = W - PAD_L - PAD_R
PLOT_H = H - PAD_T - PAD_B

def nice_ticks(vmin, vmax, n=5):
    """Generate nice tick values."""
    if vmax <= vmin:
        vmax = vmin + 1
    raw = (vmax - vmin) / n
    mag = 10 ** math.floor(math.log10(raw))
    residual = raw / mag
    if residual <= 1.5:
        nice = 1
    elif residual <= 3:
        nice = 2
    elif residual <= 7:
        nice = 5
    else:
        nice = 10
    step = nice * mag
    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step
    ticks = []
    v = lo
    while v <= hi + step * 0.01:
        ticks.append(round(v, 10))
        v += step
    return ticks, lo, hi


def fmt_val(v):
    if v >= 10000:
        return f"{v/1000:.0f}K"
    elif v >= 1000:
        return f"{v:,.0f}"
    elif v >= 1:
        return f"{v:.1f}"
    else:
        return f"{v:.2f}"


def make_chart(title, y_label, series, filename, use_log_y=False):
    """
    series: list of (name, [values]) where values correspond to DEPTHS
    """
    # Compute y range
    all_vals = [v for _, vals in series for v in vals]
    vmin_data = min(all_vals)
    vmax_data = max(all_vals)

    if use_log_y:
        # Log scale
        log_min = math.floor(math.log10(max(vmin_data, 0.1)))
        log_max = math.ceil(math.log10(vmax_data))
        y_lo, y_hi = 10**log_min, 10**log_max
        ticks_y = [10**i for i in range(log_min, log_max + 1)]

        def y_map(v):
            if v <= 0:
                v = 0.1
            return PAD_T + PLOT_H - (math.log10(v) - math.log10(y_lo)) / (math.log10(y_hi) - math.log10(y_lo)) * PLOT_H
    else:
        ticks_y, y_lo, y_hi = nice_ticks(0, vmax_data * 1.05)
        y_lo = 0

        def y_map(v):
            return PAD_T + PLOT_H - (v - y_lo) / (y_hi - y_lo) * PLOT_H

    # X positions (evenly spaced)
    n = len(DEPTHS)
    x_positions = [PAD_L + i * PLOT_W / (n - 1) for i in range(n)]

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui, -apple-system, sans-serif">')
    # Background
    svg.append(f'<rect width="{W}" height="{H}" fill="#fafafa" rx="8"/>')

    # Title
    svg.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="16" font-weight="600" fill="#1a1a1a">{title}</text>')

    # Grid & Y axis
    for t in ticks_y:
        if use_log_y and t < y_lo:
            continue
        y = y_map(t)
        if PAD_T - 5 <= y <= PAD_T + PLOT_H + 5:
            svg.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L+PLOT_W}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
            svg.append(f'<text x="{PAD_L-10}" y="{y:.1f}" text-anchor="end" dominant-baseline="middle" font-size="12" fill="#6b7280">{fmt_val(t)}</text>')

    # Y label
    svg.append(f'<text x="16" y="{PAD_T + PLOT_H/2}" text-anchor="middle" dominant-baseline="middle" font-size="13" fill="#4b5563" transform="rotate(-90,16,{PAD_T + PLOT_H/2})">{y_label}</text>')

    # X axis
    for i, (xp, label) in enumerate(zip(x_positions, DEPTH_LABELS)):
        svg.append(f'<line x1="{xp:.1f}" y1="{PAD_T}" x2="{xp:.1f}" y2="{PAD_T+PLOT_H}" stroke="#f3f4f6" stroke-width="1"/>')
        svg.append(f'<text x="{xp:.1f}" y="{PAD_T+PLOT_H+20}" text-anchor="middle" font-size="12" fill="#6b7280">{label}</text>')
    svg.append(f'<text x="{PAD_L + PLOT_W/2}" y="{H-10}" text-anchor="middle" font-size="13" fill="#4b5563">Context Depth</text>')

    # Plot area border
    svg.append(f'<rect x="{PAD_L}" y="{PAD_T}" width="{PLOT_W}" height="{PLOT_H}" fill="none" stroke="#d1d5db" stroke-width="1"/>')

    # Lines and dots
    for name, vals in series:
        color = COLORS[name]
        dash = DASH[name]
        points = []
        for i, v in enumerate(vals):
            x = x_positions[i]
            y = y_map(v)
            y = max(PAD_T, min(PAD_T + PLOT_H, y))
            points.append((x, y))

        # Line
        path = " ".join(f"{'M' if i==0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"{dash_attr}/>')

        # Dots
        for x, y in points:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="white" stroke-width="1.5"/>')

    # Legend
    legend_x = PAD_L + 10
    legend_y = PAD_T + 10
    box_w = 200
    box_h = len(series) * 22 + 10
    svg.append(f'<rect x="{legend_x}" y="{legend_y}" width="{box_w}" height="{box_h}" fill="white" fill-opacity="0.9" stroke="#e5e7eb" rx="4"/>')
    for i, (name, _) in enumerate(series):
        ly = legend_y + 20 + i * 22
        color = COLORS[name]
        dash = DASH[name]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<line x1="{legend_x+10}" y1="{ly}" x2="{legend_x+35}" y2="{ly}" stroke="{color}" stroke-width="2.5"{dash_attr}/>')
        svg.append(f'<circle cx="{legend_x+22}" cy="{ly}" r="3" fill="{color}"/>')
        svg.append(f'<text x="{legend_x+42}" y="{ly}" dominant-baseline="middle" font-size="12" fill="#374151">{name}</text>')

    svg.append('</svg>')
    with open(filename, 'w') as f:
        f.write('\n'.join(svg))
    print(f"  Created {filename}")


# ===== Generate charts =====

print("Generating charts...")

# 1. Prompt Processing Throughput (pp2048 t/s)
make_chart(
    "Prompt Processing Throughput (pp2048)",
    "Tokens/sec",
    [
        ("C1 Marlin",     c1_marlin["pp_ts"]),
        ("C1 FlashInfer", c1_flash["pp_ts"]),
        ("C4 Marlin",     c4_marlin["pp_ts"]),
        ("C4 FlashInfer", c4_flash["pp_ts"]),
    ],
    "charts/01_pp_throughput.svg"
)

# 2. Text Generation Speed (tg32 t/s total)
make_chart(
    "Text Generation Speed — Total (tg32)",
    "Tokens/sec",
    [
        ("C1 Marlin",     c1_marlin["tg_ts"]),
        ("C1 FlashInfer", c1_flash["tg_ts"]),
        ("C4 Marlin",     c4_marlin["tg_ts"]),
        ("C4 FlashInfer", c4_flash["tg_ts"]),
    ],
    "charts/02_tg_total_throughput.svg"
)

# 3. Text Generation Peak Speed (tg32 peak t/s)
make_chart(
    "Text Generation Peak Speed (tg32)",
    "Tokens/sec",
    [
        ("C1 Marlin",     c1_marlin["tg_peak"]),
        ("C1 FlashInfer", c1_flash["tg_peak"]),
        ("C4 Marlin",     c4_marlin["tg_peak"]),
        ("C4 FlashInfer", c4_flash["tg_peak"]),
    ],
    "charts/03_tg_peak_speed.svg"
)

# 4. Time to First Response (ttfr ms) - log scale
make_chart(
    "Time to First Response (pp2048)",
    "Latency (ms)",
    [
        ("C1 Marlin",     c1_marlin["ttfr"]),
        ("C1 FlashInfer", c1_flash["ttfr"]),
        ("C4 Marlin",     c4_marlin["ttfr"]),
        ("C4 FlashInfer", c4_flash["ttfr"]),
    ],
    "charts/04_ttfr.svg",
    use_log_y=True
)

# 5. E2E Time to First Token (ms) - log scale
make_chart(
    "End-to-End Time to First Token (pp2048)",
    "Latency (ms)",
    [
        ("C1 Marlin",     c1_marlin["e2e_ttft"]),
        ("C1 FlashInfer", c1_flash["e2e_ttft"]),
        ("C4 Marlin",     c4_marlin["e2e_ttft"]),
        ("C4 FlashInfer", c4_flash["e2e_ttft"]),
    ],
    "charts/05_e2e_ttft.svg",
    use_log_y=True
)

# 6. Per-Request Generation Speed (tg32 t/s per request, C4 only + C1 for reference)
make_chart(
    "Per-Request Generation Speed (tg32)",
    "Tokens/sec/req",
    [
        ("C1 Marlin",     c1_marlin["tg_ts"]),
        ("C1 FlashInfer", c1_flash["tg_ts"]),
        ("C4 Marlin",     c4_marlin["tg_ts_req"]),
        ("C4 FlashInfer", c4_flash["tg_ts_req"]),
    ],
    "charts/06_tg_per_req.svg"
)

# 7. Per-Request Prompt Processing (C4 per-req + C1 for reference)
make_chart(
    "Per-Request Prompt Processing (pp2048)",
    "Tokens/sec/req",
    [
        ("C1 Marlin",     c1_marlin["pp_ts"]),
        ("C1 FlashInfer", c1_flash["pp_ts"]),
        ("C4 Marlin",     c4_marlin["pp_ts_req"]),
        ("C4 FlashInfer", c4_flash["pp_ts_req"]),
    ],
    "charts/07_pp_per_req.svg"
)

print("Done!")
