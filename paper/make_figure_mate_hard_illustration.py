#!/usr/bin/env python3
"""Make figure illustrating MATE vs MATE-Hard.

Four panels on a 2x2 grid:
  (a) plain MATE: uniform cameras, targets, warehouses, obstacles
  (b) + heterogeneous cameras: 3 types with distinct FoV shapes & colors
  (c) + energy budgets: per-camera energy gauges, charging station
  (d) + dynamic fog: drifting occluder cloud blocks part of the FoV

All four panels share the same 2000x2000 terrain layout for direct comparison.
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, Rectangle, Circle, FancyBboxPatch, RegularPolygon
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "figures", "mate_vs_matehard.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
TERRAIN = 2000.0

# Shared layout (kept identical across panels for visual continuity)
CAMS = [
    dict(pos=(400, 500),  heading=20,   fov=70, range_=550),
    dict(pos=(1100, 350), heading=110,  fov=70, range_=550),
    dict(pos=(900, 1500), heading=-70,  fov=70, range_=550),
    dict(pos=(1650, 1300), heading=180, fov=70, range_=550),
]
TARGETS = [(700, 800), (1300, 900), (1500, 600), (900, 1100), (600, 1300), (1700, 1000)]
WAREHOUSES = [(200, 200), (1800, 200), (200, 1800), (1800, 1800)]
OBSTACLES = [
    dict(xy=(1050, 950), r=120),
    dict(xy=(550, 1100), r=90),
]

CAM_TYPES = ["wide", "tele", "fisheye"]  # 3 heterogeneous types
TYPE_COLORS = {"wide": "#4C9AFF", "tele": "#FF8C42", "fisheye": "#9B5DE5"}
TYPE_FOV    = {"wide": 110, "tele": 40, "fisheye": 200}
TYPE_RANGE  = {"wide": 480, "tele": 700, "fisheye": 360}

# Per-camera type assignment (only relevant for panels b,c,d)
CAM_TYPE_ASSIGN = ["wide", "tele", "wide", "fisheye"]

ENERGY = [0.95, 0.35, 0.70, 0.15]  # cam 1 and cam 3 are low
CHARGER = (1750, 1750)
FOG_POLY = np.array([(750, 350), (1500, 250), (1700, 1100), (1100, 1250), (700, 950)])


def setup_axes(ax, title):
    ax.set_xlim(0, TERRAIN)
    ax.set_ylim(0, TERRAIN)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#F7F5EF")
    for spine in ax.spines.values():
        spine.set_color("#888")
        spine.set_linewidth(0.8)
    ax.set_title(title, fontsize=11, pad=6, loc="left", fontweight="bold")


def draw_warehouses(ax):
    for (x, y) in WAREHOUSES:
        ax.add_patch(Rectangle((x - 70, y - 70), 140, 140,
                               facecolor="#D9C9A3", edgecolor="#7A6A40", linewidth=1.0, zorder=1))


def draw_obstacles(ax):
    for o in OBSTACLES:
        ax.add_patch(Circle(o["xy"], o["r"], facecolor="#5C5C5C", edgecolor="#2A2A2A", linewidth=0.5, zorder=1.5))


def draw_targets(ax):
    for (x, y) in TARGETS:
        ax.plot(x, y, marker="o", markersize=7, markerfacecolor="#E63946",
                markeredgecolor="#2A2A2A", markeredgewidth=0.6, linestyle="None", zorder=4)


def fov_wedge(pos, heading, fov, range_, color, alpha=0.30, edge=None):
    theta1 = heading - fov / 2
    theta2 = heading + fov / 2
    return Wedge(pos, range_, theta1, theta2,
                 facecolor=color, edgecolor=edge or color, alpha=alpha, linewidth=0.8, zorder=2)


def draw_camera_body(ax, pos, color, energy_frac=None):
    cx, cy = pos
    ax.add_patch(Circle((cx, cy), 38, facecolor=color, edgecolor="#1A1A1A", linewidth=0.8, zorder=5))
    if energy_frac is not None:
        # small horizontal energy gauge above the camera
        gw, gh = 80, 14
        gx, gy = cx - gw / 2, cy + 55
        ax.add_patch(Rectangle((gx, gy), gw, gh, facecolor="white",
                               edgecolor="#222", linewidth=0.6, zorder=6))
        fc = "#2ECC71" if energy_frac > 0.5 else ("#F1C40F" if energy_frac > 0.25 else "#E74C3C")
        ax.add_patch(Rectangle((gx + 1, gy + 1), max(1, (gw - 2) * energy_frac), gh - 2,
                               facecolor=fc, edgecolor="none", zorder=6.5))


def panel_a_plain(ax):
    """Plain MATE: identical cameras, identical FoVs, no fog, no energy."""
    setup_axes(ax, "(a) plain MATE")
    draw_warehouses(ax); draw_obstacles(ax)
    for cam in CAMS:
        ax.add_patch(fov_wedge(cam["pos"], cam["heading"], cam["fov"], cam["range_"], "#4C9AFF", alpha=0.32))
    for cam in CAMS:
        draw_camera_body(ax, cam["pos"], "#4C9AFF")
    draw_targets(ax)


def panel_b_het(ax):
    """+ heterogeneous cameras: 3 types with distinct FoV widths/ranges/colors."""
    setup_axes(ax, "(b) + heterogeneous cameras")
    draw_warehouses(ax); draw_obstacles(ax)
    for cam, ctype in zip(CAMS, CAM_TYPE_ASSIGN):
        ax.add_patch(fov_wedge(cam["pos"], cam["heading"], TYPE_FOV[ctype], TYPE_RANGE[ctype],
                               TYPE_COLORS[ctype], alpha=0.30))
    for cam, ctype in zip(CAMS, CAM_TYPE_ASSIGN):
        draw_camera_body(ax, cam["pos"], TYPE_COLORS[ctype])
    draw_targets(ax)


def panel_c_energy(ax):
    """+ energy budgets: each camera has an energy gauge; charging station appears."""
    setup_axes(ax, "(c) + energy budgets")
    draw_warehouses(ax); draw_obstacles(ax)
    # charging station (hex on a pad)
    cx, cy = CHARGER
    ax.add_patch(Rectangle((cx - 90, cy - 90), 180, 180,
                           facecolor="#FFF3B0", edgecolor="#B58900", linewidth=1.0, zorder=1))
    ax.add_patch(RegularPolygon((cx, cy), numVertices=6, radius=45,
                                facecolor="#F4D35E", edgecolor="#7A5C00", linewidth=1.0, zorder=2))
    ax.text(cx, cy, "⚡", ha="center", va="center", fontsize=18,
            color="#B58900", zorder=3)
    for cam, ctype, e in zip(CAMS, CAM_TYPE_ASSIGN, ENERGY):
        col = TYPE_COLORS[ctype]
        # dim the FoV if low energy
        alpha = 0.10 if e < 0.25 else (0.20 if e < 0.5 else 0.30)
        ax.add_patch(fov_wedge(cam["pos"], cam["heading"], TYPE_FOV[ctype], TYPE_RANGE[ctype],
                               col, alpha=alpha))
    for cam, ctype, e in zip(CAMS, CAM_TYPE_ASSIGN, ENERGY):
        draw_camera_body(ax, cam["pos"], TYPE_COLORS[ctype], energy_frac=e)
    draw_targets(ax)


def panel_d_fog(ax):
    """+ dynamic fog: a drifting cloud occludes part of the terrain."""
    setup_axes(ax, "(d) + dynamic fog")
    draw_warehouses(ax); draw_obstacles(ax)
    for cam, ctype in zip(CAMS, CAM_TYPE_ASSIGN):
        ax.add_patch(fov_wedge(cam["pos"], cam["heading"], TYPE_FOV[ctype], TYPE_RANGE[ctype],
                               TYPE_COLORS[ctype], alpha=0.30))
    for cam, ctype in zip(CAMS, CAM_TYPE_ASSIGN):
        draw_camera_body(ax, cam["pos"], TYPE_COLORS[ctype])
    draw_targets(ax)
    # fog cloud on top (semi-transparent layer)
    fog = mpatches.Polygon(FOG_POLY, closed=True, facecolor="#B0BEC5", alpha=0.65,
                           edgecolor="#546E7A", linewidth=0.8, zorder=3.5)
    ax.add_patch(fog)
    # drift arrow
    fc = FOG_POLY.mean(axis=0)
    ax.annotate("", xy=(fc[0] + 250, fc[1] + 120), xytext=(fc[0], fc[1]),
                arrowprops=dict(arrowstyle="->", color="#37474F", lw=1.6), zorder=4)
    ax.text(fc[0] + 120, fc[1] + 220, "drift", color="#37474F", fontsize=9, ha="center")


def build_legend(fig):
    handles = [
        mpatches.Patch(facecolor="#4C9AFF", edgecolor="#1A1A1A", label="wide camera"),
        mpatches.Patch(facecolor="#FF8C42", edgecolor="#1A1A1A", label="tele camera"),
        mpatches.Patch(facecolor="#9B5DE5", edgecolor="#1A1A1A", label="fisheye camera"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#E63946",
                   markeredgecolor="#2A2A2A", markersize=7, label="target"),
        mpatches.Patch(facecolor="#D9C9A3", edgecolor="#7A6A40", label="warehouse"),
        mpatches.Patch(facecolor="#5C5C5C", edgecolor="#2A2A2A", label="static obstacle"),
        mpatches.Patch(facecolor="#F4D35E", edgecolor="#7A5C00", label="charging station"),
        mpatches.Patch(facecolor="#B0BEC5", edgecolor="#546E7A", alpha=0.65, label="fog (occluder)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.4))
    panel_a_plain(axes[0, 0])
    panel_b_het(axes[0, 1])
    panel_c_energy(axes[1, 0])
    panel_d_fog(axes[1, 1])
    build_legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    fig.savefig(OUT.replace(".pdf", ".png"), bbox_inches="tight", dpi=180)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
