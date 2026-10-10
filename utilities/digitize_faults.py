# -*- coding: utf-8 -*-
"""
Interactive digitizer for tectonic-block polygons.
==================================================

The block division of the study area follows the major fault zones and basement
boundaries that are recognized in regional geological studies. This script is the
tool that was used to draw those blocks: the fault traces and the outline of the
study area are plotted, and one closed polygon is drawn per block with the mouse.

Every vertex of the polygon drawn by hand is snapped to the nearest fault trace or
to the study-area outline, so that the blocks are bounded by mapped geological
features rather than by arbitrary lines. Each polygon is then written to
``block_<N>.txt`` and later cleaned by ``clean_blocks.py``.

Controls
--------
* hold the left mouse button and drag to draw one block, then release
* press ``u`` to undo the last block (its ``block_<N>.txt`` file is deleted)
* press ``q`` to quit

Usage
-----
    python utilities/digitize_faults.py --faults data/fault_traces.txt \
        --outdir blocks --lon-min 100 --lon-max 118 --lat-min 31 --lat-max 43
"""

import argparse
import os

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union, nearest_points


# ======================================================================
# 1. Read a multi-line fault file
# ======================================================================

def read_multiline_txt(path):
    """Read a multi-polyline text file and return a GeoDataFrame (EPSG:4326).

    Each polyline starts with a header line beginning with ``>``; the following
    lines hold one ``lon lat`` (or ``lon, lat``) pair per vertex. A header may be
    omitted for the first polyline.
    """
    lines = []
    coords = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith('>'):
                if coords:
                    lines.append(LineString(coords))
                    coords = []
                continue
            parts = [p for p in s.replace(',', ' ').split() if p]
            if len(parts) < 2:
                continue
            lon, lat = map(float, parts[:2])
            coords.append((lon, lat))
    if coords:
        lines.append(LineString(coords))
    return gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")


def main():
    parser = argparse.ArgumentParser(
        description="Interactively digitize tectonic-block polygons that snap to fault traces."
    )
    parser.add_argument("--faults", type=str, default="data/fault_traces.txt",
                        help="text file with the fault traces (default: data/fault_traces.txt)")
    parser.add_argument("--outdir", type=str, default="blocks",
                        help="directory in which block_<N>.txt files are written "
                             "(default: blocks)")
    parser.add_argument("--lon-min", type=float, default=100.0, help="western bound of the study area")
    parser.add_argument("--lon-max", type=float, default=118.0, help="eastern bound of the study area")
    parser.add_argument("--lat-min", type=float, default=31.0, help="southern bound of the study area")
    parser.add_argument("--lat-max", type=float, default=43.0, help="northern bound of the study area")
    parser.add_argument("--start-id", type=int, default=1,
                        help="number given to the first block drawn (default: 1)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    lon_min, lon_max = args.lon_min, args.lon_max
    lat_min, lat_max = args.lat_min, args.lat_max

    # ==================================================================
    # 2. Load the fault traces and build the study-area outline
    # ==================================================================
    fault_lines = read_multiline_txt(args.faults)
    print(f"Loaded {len(fault_lines)} fault trace(s) from {args.faults}")

    outer_poly = Polygon([
        (lon_min, lat_min),
        (lon_max, lat_min),
        (lon_max, lat_max),
        (lon_min, lat_max),
        (lon_min, lat_min),
    ])
    outer_boundary = outer_poly.boundary

    # Faults and the outline are merged into a single geometry, which is the set of
    # lines onto which the hand-drawn vertices are snapped.
    all_lines = list(fault_lines.geometry) + [outer_boundary]
    union_lines = unary_union(all_lines)

    # ==================================================================
    # 3. Plot the base map
    # ==================================================================
    fig, ax = plt.subplots(figsize=(8, 8))
    fault_lines.plot(ax=ax, color="red", linewidth=0.8, label="Faults")
    gpd.GeoSeries([outer_boundary], crs="EPSG:4326").plot(
        ax=ax, linestyle="--", color="black", linewidth=1.0, label="Study-area outline"
    )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    ax.set_title("Draw one block with the left mouse button; release to save.\n"
                 "Press u to undo the last block, q to quit.")
    ax.set_aspect("equal", adjustable="box")

    block_id = args.start_id
    # (line artist, text artist, file name) of the blocks drawn so far
    drawn_blocks = []

    # ==================================================================
    # 4. Lasso callback: one stroke defines one block
    # ==================================================================

    def onselect(verts):
        """``verts`` holds the vertices of the path drawn by the user."""
        nonlocal block_id

        # (1) snap every vertex to the nearest fault trace or outline
        snapped = []
        for x, y in verts:
            p = Point(x, y)
            _, p_on_line = nearest_points(p, union_lines)
            snapped.append((p_on_line.x, p_on_line.y))

        if len(snapped) < 3:
            print("Too few vertices to form a polygon; this stroke was ignored.")
            return

        # (2) build the polygon (closed automatically)
        poly = Polygon(snapped)

        # (3) display it
        xs, ys = poly.exterior.xy
        line_artist, = ax.plot(xs, ys, color="orange", linewidth=1.5)
        cx, cy = poly.representative_point().x, poly.representative_point().y
        text_artist = ax.text(cx, cy, str(block_id), color="purple",
                              ha="center", va="center", fontsize=12)
        fig.canvas.draw_idle()

        # (4) write the polygon to block_<N>.txt
        out_name = os.path.join(args.outdir, f"block_{block_id}.txt")
        with open(out_name, "w", encoding="utf-8") as f:
            for x, y in poly.exterior.coords:
                f.write(f"{x:.6f} {y:.6f}\n")
        print(f"Saved block {block_id} to {out_name}")

        drawn_blocks.append((line_artist, text_artist, out_name))
        block_id += 1

    lasso = LassoSelector(ax, onselect=onselect)  # noqa: F841

    # ==================================================================
    # 5. Key bindings: u to undo, q to quit
    # ==================================================================

    def on_key(event):
        nonlocal block_id

        if event.key == "u":
            if not drawn_blocks:
                print("Nothing to undo.")
                return
            line_artist, text_artist, filename = drawn_blocks.pop()
            line_artist.remove()
            text_artist.remove()
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Undone and deleted {filename}")
            else:
                print(f"Block undone, but {filename} was not found")
            block_id -= 1
            fig.canvas.draw_idle()

        elif event.key == "q":
            print("Exiting.")
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.legend(loc="lower left")
    plt.show()


if __name__ == "__main__":
    main()
