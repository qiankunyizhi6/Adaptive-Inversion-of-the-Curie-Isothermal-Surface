# -*- coding: utf-8 -*-
"""
Clean-up of the digitized tectonic-block polygons.
=================================================

The polygons produced by ``digitize_faults.py`` are drawn by hand, so they may
overlap slightly, leave narrow gaps between neighbouring blocks, or be invalid
(self-intersecting) geometries. This script repairs them before the inversion:

1. read every ``block_<N>*.txt`` polygon;
2. validate the geometry;
3. remove overlaps, giving priority to the lower block number;
4. assign every remaining gap to the nearest block;
5. force each geometry to a single polygon;
6. write the repaired polygons as ``block_<N>_fixed.txt``.

The repaired files are the ones read by
``curie_block_bayes_three_class_prior_inversion.py``.

Usage
-----
    python utilities/clean_blocks.py --blockdir blocks \
        --pattern "block_*_fixed.txt" --plot
"""

import argparse
import glob
import os
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union
from shapely.validation import make_valid


# ======================================================================
# 1. Read one block file
# ======================================================================

def read_block_txt(path):
    """Read a polygon file with one ``lon lat`` pair per line."""
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            parts = [p for p in s.replace(",", " ").split() if p]
            if len(parts) < 2:
                continue
            lon, lat = map(float, parts[:2])
            xs.append(lon)
            ys.append(lat)
    if len(xs) < 3:
        raise ValueError(f"{path}: fewer than 3 vertices, cannot form a polygon")
    return np.column_stack([xs, ys])


# ======================================================================
# 2. Geometry helpers
# ======================================================================

def clean_geom(g):
    """Return a valid version of ``g`` (make_valid, falling back to buffer(0))."""
    if g.is_empty:
        return g
    try:
        return make_valid(g)
    except Exception:
        return g.buffer(0)


def force_polygon(g):
    """Coerce ``g`` to a single Polygon, or return None if that is impossible."""
    g = clean_geom(g)
    if g.is_empty:
        return None

    if g.geom_type == "Polygon":
        return g

    if g.geom_type == "MultiPolygon":
        return max(list(g.geoms), key=lambda p: p.area)

    # LineString / MultiLineString / GeometryCollection: try to polygonize first
    try:
        polys = list(polygonize(g))
    except Exception:
        polys = []
    if polys:
        return max(polys, key=lambda p: p.area)

    # Last resort: buffer(0)
    buf = g.buffer(0)
    if buf.geom_type == "Polygon":
        return buf
    if buf.geom_type == "MultiPolygon":
        return max(list(buf.geoms), key=lambda p: p.area)

    return None


def block_id_from_name(path):
    """Extract the integer block number from a file name such as block_7_fixed.txt."""
    m = re.search(r"block[_-]?(\d+)", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot infer a block number from {path}")
    return int(m.group(1))


# ----------------------------------------------------------------------
# Robust set operations
#
# Hand-digitized polygons often touch along shared boundaries and leave
# sub-metre slivers, which makes GEOS raise a TopologyException. The
# helpers below retry with a small buffer and, if available, with a
# grid-snapped operation, so that the cleaning step is reproducible
# across shapely / GEOS versions.
# ----------------------------------------------------------------------

SNAP_GRID = 1e-6      # degrees, about 0.1 m
SLIVER_BUFFER = 1e-9  # degrees


def _grid_snap(g):
    """Snap a geometry to a fine grid (shapely 2.x only)."""
    fn = getattr(shapely, "set_precision", None)
    if fn is None:
        return None
    try:
        snapped = fn(g, grid_size=SNAP_GRID)
        if snapped is not None and not snapped.is_empty:
            return snapped
    except Exception:
        pass
    return None


def safe_difference(a, b):
    """Difference ``a - b`` with graceful fallbacks."""
    last_exc = None
    for op in (
        lambda: a.difference(b),
        lambda: clean_geom(a).difference(clean_geom(b)),
        lambda: clean_geom(a).difference(clean_geom(b).buffer(SLIVER_BUFFER)),
    ):
        try:
            return op()
        except Exception as exc:
            last_exc = exc

    ga, gb = _grid_snap(clean_geom(a)), _grid_snap(clean_geom(b))
    if ga is not None and gb is not None:
        try:
            return ga.difference(gb)
        except Exception as exc:
            last_exc = exc

    print(f"    difference could not be resolved ({last_exc}); the overlapped part is kept")
    return clean_geom(a)


def safe_union(geoms):
    """Union of several geometries with graceful fallbacks."""
    last_exc = None
    for op in (
        lambda: unary_union(list(geoms)),
        lambda: unary_union([clean_geom(g) for g in geoms]),
        lambda: unary_union([clean_geom(g).buffer(SLIVER_BUFFER) for g in geoms]),
    ):
        try:
            return op()
        except Exception as exc:
            last_exc = exc

    snapped = [_grid_snap(clean_geom(g)) for g in geoms]
    if all(s is not None for s in snapped):
        try:
            return unary_union(snapped)
        except Exception as exc:
            last_exc = exc

    print(f"    union could not be resolved ({last_exc}); geometries are stacked in order")
    out = None
    for g in geoms:
        out = clean_geom(g) if out is None else out.union(clean_geom(g))
    return out



# ======================================================================
# 3. Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Remove overlaps and fill gaps in the digitized tectonic-block polygons."
    )
    parser.add_argument("--blockdir", type=str, default="blocks",
                        help="directory holding the block polygon files (default: blocks)")
    parser.add_argument("--pattern", type=str, default="block_*_fixed.txt",
                        help="glob pattern of the input files (default: block_*_fixed.txt)")
    parser.add_argument("--out-suffix", type=str, default="_fixed",
                        help="suffix appended to the output file names; files already "
                             "ending with it are overwritten in place (default: _fixed)")
    parser.add_argument("--plot", action="store_true",
                        help="show a check plot of the repaired blocks before writing")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.blockdir, args.pattern)),
                   key=block_id_from_name)
    if not files:
        raise SystemExit(f"No files matched {args.pattern} in {args.blockdir}")
    print(f"Found {len(files)} block file(s).")

    # ==================================================================
    # 4. Load the polygons
    # ==================================================================
    geoms, ids = [], []
    for fname in files:
        coords = read_block_txt(fname)
        geoms.append(clean_geom(Polygon(coords)))
        ids.append(block_id_from_name(fname))

    blocks = gpd.GeoDataFrame({"Block_ID": ids, "geometry": geoms}, crs="EPSG:4326")
    print("Blocks read:", len(blocks))

    # ==================================================================
    # 5. Remove overlaps: the lower block number keeps the shared area
    # ==================================================================
    blocks = blocks.sort_values("Block_ID").reset_index(drop=True)

    clean_geoms = []
    current_union = None
    for _, row in blocks.iterrows():
        geom = clean_geom(row.geometry)

        if current_union is not None and not current_union.is_empty:
            geom = safe_difference(geom, clean_geom(current_union))

        clean_geoms.append(geom)

        if current_union is None or current_union.is_empty:
            current_union = geom
        else:
            current_union = safe_union([clean_geom(current_union), clean_geom(geom)])

    blocks["geometry"] = clean_geoms
    blocks = blocks[~blocks.geometry.is_empty].copy().reset_index(drop=True)
    print("Valid blocks after overlap removal:", len(blocks))

    # ==================================================================
    # 6. Fill the gaps: each gap goes to the nearest block
    # ==================================================================
    all_bounds = np.array([g.bounds for g in blocks.geometry])
    minx, miny = all_bounds[:, 0].min(), all_bounds[:, 1].min()
    maxx, maxy = all_bounds[:, 2].max(), all_bounds[:, 3].max()

    outer_poly = clean_geom(Polygon([
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
        (minx, miny),
    ]))

    union_clean = safe_union([clean_geom(g) for g in blocks.geometry])
    gap_geom = safe_difference(outer_poly, clean_geom(union_clean))

    gap_polys = []
    if gap_geom.is_empty:
        print("No gap to fill.")
    elif gap_geom.geom_type == "Polygon":
        gap_polys = [gap_geom]
    elif gap_geom.geom_type == "MultiPolygon":
        gap_polys = list(gap_geom.geoms)
    elif isinstance(gap_geom, shapely.geometry.collection.GeometryCollection):
        for gg in gap_geom.geoms:
            if gg.geom_type == "Polygon":
                gap_polys.append(gg)
            elif gg.geom_type == "MultiPolygon":
                gap_polys.extend(list(gg.geoms))

    print("Gap polygons:", len(gap_polys))

    for gap in gap_polys:
        gap = clean_geom(gap)
        cen = gap.representative_point()
        distances = [cen.distance(clean_geom(g)) for g in blocks.geometry]
        nearest_idx = int(np.argmin(distances))
        blocks.at[nearest_idx, "geometry"] = safe_union(
            [clean_geom(blocks.geometry.iloc[nearest_idx]), gap]
        )

    blocks = blocks.dissolve(by="Block_ID", as_index=False)
    print("Blocks after gap filling:", len(blocks))

    # ==================================================================
    # 7. Force every geometry to a Polygon
    # ==================================================================
    blocks["geometry"] = blocks["geometry"].apply(force_polygon)
    blocks = blocks[blocks.geometry.notnull()].copy().reset_index(drop=True)
    print("Blocks after conversion to Polygon:", len(blocks))

    # ==================================================================
    # 8. Optional check plot
    # ==================================================================
    if args.plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                  "#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]

        for _, row in blocks.iterrows():
            geom, bid = row.geometry, int(row["Block_ID"])
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, facecolor=colors[(bid - 1) % len(colors)],
                    edgecolor="black", alpha=0.5, linewidth=1.0)
            rp = geom.representative_point()
            ax.text(rp.x, rp.y, str(bid), ha="center", va="center",
                    fontsize=12, color="black")

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude (deg E)")
        ax.set_ylabel("Latitude (deg N)")
        ax.set_title("Tectonic blocks after cleaning")
        plt.tight_layout()
        plt.show()

    # ==================================================================
    # 9. Write the repaired polygons
    # ==================================================================
    for _, row in blocks.iterrows():
        bid = int(row["Block_ID"])
        poly = force_polygon(row.geometry)
        if poly is None:
            print(f"Block {bid} could not be converted to a polygon; skipped.")
            continue

        out_name = os.path.join(args.blockdir, f"block_{bid}{args.out_suffix}.txt")
        with open(out_name, "w", encoding="utf-8") as f:
            for x, y in poly.exterior.coords:
                f.write(f"{x:.6f} {y:.6f}\n")
        print(f"Wrote repaired block {bid} to {out_name}")


if __name__ == "__main__":
    main()
