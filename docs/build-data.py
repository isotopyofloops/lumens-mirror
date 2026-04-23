#!/usr/bin/env python3
"""
Build Lumen's mirror data.json from cluster + embedding data.

Uses 868 individual files as nodes, grouped into color clusters
via agglomerative clustering on embeddings. Edges from cosine
similarity between files above threshold.
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

CLUSTERS_PATH = Path(os.path.expanduser(
    "~/autonomous-ai/connection-map-public/docs/lumen-clusters.json"
))
EMBEDDINGS_PATH = Path(os.path.expanduser(
    "~/autonomous-ai/connection-map-public/docs/lumen-embeddings.json"
))
OUTPUT = Path(__file__).parent / "data.json"

# Loose threshold for color grouping (~10-20 groups)
COLOR_CLUSTER_THRESHOLD = 0.66
# Similarity threshold for drawing edges
EDGE_SIM_THRESHOLD = 0.65
MAX_EDGES = 2000
TOP_NEIGHBORS = 8

CLUSTER_COLORS = [
    "#64b5f6",  # blue
    "#ff8a65",  # coral
    "#81c784",  # green
    "#ba68c8",  # purple
    "#ffb74d",  # amber
    "#4dd0e1",  # cyan
    "#f06292",  # pink
    "#aed581",  # lime
    "#7986cb",  # indigo
    "#e57373",  # red
    "#4db6ac",  # teal
    "#fff176",  # yellow
    "#90a4ae",  # blue-grey
    "#ce93d8",  # lavender
    "#a1887f",  # brown
    "#80cbc4",  # mint
]


def main():
    print("Loading cluster data...")
    with open(CLUSTERS_PATH) as f:
        cluster_data = json.load(f)

    print("Loading embeddings...")
    with open(EMBEDDINGS_PATH) as f:
        emb_cache = json.load(f)

    # Flatten all files from clusters, preserving cluster membership
    files = []
    fine_cluster_map = {}
    for cluster in cluster_data["clusters"]:
        cluster_name = cluster["representative"]
        for finfo in cluster["files"]:
            files.append(finfo)
            fine_cluster_map[finfo["source_url"]] = cluster_name

    print(f"Total files: {len(files)}")

    # Build embedding matrix
    valid_files = []
    emb_list = []
    for f in files:
        url = f["source_url"]
        if url in emb_cache:
            valid_files.append(f)
            emb_list.append(emb_cache[url])

    embeddings = np.array(emb_list)
    print(f"Files with embeddings: {len(valid_files)}")

    # Normalize for cosine
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = embeddings / norms
    sim = normed @ normed.T

    # Color clustering (loose, for ~10-20 groups)
    print("Computing color clusters...")
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    condensed = squareform(dist)
    Z = linkage(condensed, method="average")
    color_labels = fcluster(Z, t=COLOR_CLUSTER_THRESHOLD, criterion="distance")

    # Name color clusters by their largest member's fine cluster
    color_groups = {}
    for i, label in enumerate(color_labels):
        color_groups.setdefault(int(label), []).append(i)

    sorted_color_groups = sorted(color_groups.values(), key=len, reverse=True)

    node_color_cluster = {}
    node_color = {}
    for ci, members in enumerate(sorted_color_groups):
        # Find representative via centroid
        cluster_embs = embeddings[members]
        centroid = cluster_embs.mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) or 1)
        scores = cluster_embs @ centroid_norm
        rep_idx = members[np.argmax(scores)]
        name = valid_files[rep_idx]["title"]

        color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
        for m in members:
            node_color_cluster[m] = name
            node_color[m] = color

    unique_clusters = {}
    for i in range(len(valid_files)):
        c = node_color_cluster[i]
        if c not in unique_clusters:
            unique_clusters[c] = node_color[i]

    print(f"Color clusters: {len(unique_clusters)}")

    # Compute edges from similarity
    print("Computing edges...")
    edges = []
    for i in range(len(valid_files)):
        for j in range(i + 1, len(valid_files)):
            s = float(sim[i, j])
            if s >= EDGE_SIM_THRESHOLD:
                edges.append((i, j, s))

    edges.sort(key=lambda x: x[2], reverse=True)
    if len(edges) > MAX_EDGES:
        edges = edges[:MAX_EDGES]
    print(f"Edges above {EDGE_SIM_THRESHOLD}: {len(edges)}")

    # Compute semantic neighbors
    print("Computing semantic neighbors...")
    neighbors_map = {}
    for i in range(len(valid_files)):
        sims = [(j, float(sim[i, j])) for j in range(len(valid_files)) if j != i]
        sims.sort(key=lambda x: x[1], reverse=True)
        neighbors_map[i] = [
            {"id": valid_files[s[0]]["title"], "score": round(s[1], 3)}
            for s in sims[:TOP_NEIGHBORS]
        ]

    # Build output
    out_nodes = []
    for i, f in enumerate(valid_files):
        out_nodes.append({
            "id": f["title"],
            "summary": f.get("summary_preview", ""),
            "directory": f["directory"],
            "filename": f["filename"],
            "source_url": f["source_url"],
            "cluster": node_color_cluster[i],
            "color": node_color[i],
            "neighbors": neighbors_map.get(i, []),
        })

    out_links = []
    for i, j, s in edges:
        out_links.append({
            "source": valid_files[i]["title"],
            "target": valid_files[j]["title"],
            "predicate": f"similarity ({s:.0%})",
            "weight": round(s, 3),
        })

    data = {
        "nodes": out_nodes,
        "links": out_links,
        "clusters": list(unique_clusters.keys()),
        "colors": unique_clusters,
    }

    with open(OUTPUT, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nBuilt data.json: {len(out_nodes)} nodes, {len(out_links)} links, {len(unique_clusters)} clusters")


if __name__ == "__main__":
    main()
