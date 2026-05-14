#!/usr/bin/env python3
"""Convert entities.jsonl + triples.jsonl → data.json for the explorer UI.

Replaces the old build-data.py which used hardcoded cluster assignments.
This version:
  - Computes entity embeddings by averaging source-file embeddings
  - Adds cosine_similarity edges above a configurable threshold (default 0.55)
  - Uses label propagation for community detection (curated edges only)
  - Produces data.json compatible with the existing d3 UI
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).parent
GRAPH = ROOT / "graph"
DOCS = ROOT / "docs"
OUTPUT = DOCS / "data.json"

EMBEDDINGS_PATH = Path(os.path.expanduser(
    "~/autonomous-ai/connection-map-public/docs/lumen-embeddings.json"
))
GITHUB_BASE = "https://github.com/isotopyofloops/connection-sources/blob/main/lumen"

SIMILARITY_THRESHOLD = 0.70
TOP_NEIGHBORS = 8

PALETTE = [
    "#64b5f6", "#ff8a65", "#81c784", "#ba68c8", "#4dd0e1",
    "#f06292", "#ffb74d", "#7986cb", "#aed581", "#e57373",
    "#90caf9", "#ce93d8", "#a5d6a7", "#ef9a9a", "#80deea",
    "#fff176", "#bcaaa4", "#b0bec5", "#f48fb1", "#c5e1a5",
]


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def compute_embeddings(entities):
    """Average source-file embeddings per entity. Returns {name: np.array}."""
    if not EMBEDDINGS_PATH.exists():
        print(f"  Embeddings not found at {EMBEDDINGS_PATH}, skipping semantic edges")
        return {}

    with open(EMBEDDINGS_PATH) as f:
        emb_cache = json.load(f)

    entity_embs = {}
    for e in entities:
        file_embs = []
        for sf in e.get("source_files", []):
            url = f"{GITHUB_BASE}/{sf}"
            if url in emb_cache:
                file_embs.append(emb_cache[url])
        if file_embs:
            entity_embs[e["name"]] = np.mean(file_embs, axis=0)

    print(f"  Entities with embeddings: {len(entity_embs)}/{len(entities)}")
    return entity_embs


def semantic_edges(entity_embs, threshold):
    """Compute cosine similarity edges above threshold. Returns list of (src, tgt, score)."""
    if not entity_embs:
        return []

    names = sorted(entity_embs.keys())
    matrix = np.array([entity_embs[n] for n in names])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = matrix / norms
    sim = normed @ normed.T

    edges = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            score = float(sim[i, j])
            if score >= threshold:
                edges.append((names[i], names[j], score))

    return edges


def semantic_neighbors(entity_embs, top_n):
    """Compute top-N semantic neighbors per entity."""
    if not entity_embs:
        return {}

    names = sorted(entity_embs.keys())
    matrix = np.array([entity_embs[n] for n in names])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = matrix / norms
    sim = normed @ normed.T

    neighbors = {}
    for i, name in enumerate(names):
        sims = [(names[j], float(sim[i, j])) for j in range(len(names)) if j != i]
        sims.sort(key=lambda x: x[1], reverse=True)
        neighbors[name] = [{"id": s[0], "score": round(s[1], 3)} for s in sims[:top_n]]

    return neighbors


def label_propagation(adj, node_ids, seed=42):
    """Simple label propagation for community detection."""
    labels = {nid: i for i, nid in enumerate(node_ids)}
    rng = random.Random(seed)

    for _ in range(30):
        changed = False
        order = list(node_ids)
        rng.shuffle(order)
        for nid in order:
            neighbors = adj.get(nid, set())
            if not neighbors:
                continue
            neighbor_labels = [labels[nb] for nb in neighbors if nb in labels]
            if not neighbor_labels:
                continue
            counts = Counter(neighbor_labels)
            max_count = counts.most_common(1)[0][1]
            tied = [lbl for lbl, c in counts.items() if c == max_count]
            best = labels[nid] if labels[nid] in tied else min(tied)
            if labels[nid] != best:
                labels[nid] = best
                changed = True
        if not changed:
            break

    return labels


def main():
    entities = load_jsonl(GRAPH / "entities.jsonl")
    triples = load_jsonl(GRAPH / "triples.jsonl")
    entity_set = {e["name"] for e in entities}
    print(f"Loaded {len(entities)} entities, {len(triples)} triples")

    # Compute embeddings
    print("Computing embeddings...")
    entity_embs = compute_embeddings(entities)

    # Build curated edges from triples
    links = []
    for t in triples:
        if t["subject"] in entity_set and t["object"] in entity_set:
            links.append({
                "source": t["subject"],
                "target": t["object"],
                "predicate": t["predicate"],
            })
    curated_count = len(links)

    # Add semantic similarity edges above threshold
    print(f"Computing semantic edges (threshold={SIMILARITY_THRESHOLD})...")
    sem_edges = semantic_edges(entity_embs, SIMILARITY_THRESHOLD)

    curated_pairs = {(l["source"], l["target"]) for l in links}
    curated_pairs |= {(l["target"], l["source"]) for l in links}

    added = 0
    for src, tgt, score in sem_edges:
        if (src, tgt) not in curated_pairs:
            links.append({
                "source": src,
                "target": tgt,
                "predicate": "cosine_similarity",
                "score": round(score, 3),
            })
            added += 1

    print(f"  Curated edges: {curated_count}")
    print(f"  Semantic edges (>={SIMILARITY_THRESHOLD}): {added}")
    print(f"  Total edges: {len(links)}")

    # Community detection via label propagation (curated edges only)
    adj = defaultdict(set)
    for l in links:
        if l.get("predicate") == "cosine_similarity":
            continue
        adj[l["source"]].add(l["target"])
        adj[l["target"]].add(l["source"])

    node_ids = [e["name"] for e in entities]
    labels = label_propagation(adj, node_ids)

    # Rank communities by size
    community_members = defaultdict(list)
    for nid, label in labels.items():
        community_members[label].append(nid)
    ranked = sorted(community_members.items(), key=lambda x: -len(x[1]))
    remap = {old_id: new_id for new_id, (old_id, _) in enumerate(ranked)}

    community_names = {}
    for old_id, members in community_members.items():
        community_names[remap[old_id]] = members

    # Assign colors
    cluster_colors = {}
    for cid in sorted(community_names.keys()):
        cluster_colors[str(cid)] = PALETTE[cid % len(PALETTE)]

    node_community = {}
    for cid, members in community_names.items():
        for m in members:
            node_community[m] = str(cid)

    # Semantic neighbors for hover panel
    print("Computing semantic neighbors...")
    neighbors_map = semantic_neighbors(entity_embs, TOP_NEIGHBORS)

    # Build nodes
    nodes = []
    for e in entities:
        name = e["name"]
        cluster = node_community.get(name, "0")
        source_urls = [f"{GITHUB_BASE}/{sf}" for sf in e.get("source_files", [])]
        nodes.append({
            "id": name,
            "summary": e.get("summary", ""),
            "cluster": cluster,
            "color": cluster_colors.get(cluster, "#aaaaaa"),
            "neighbors": neighbors_map.get(name, []),
            "source_files": e.get("source_files", []),
            "source_urls": source_urls,
        })

    clusters_list = [str(cid) for cid in sorted(community_names.keys())]

    data = {
        "nodes": nodes,
        "links": links,
        "clusters": clusters_list,
        "colors": cluster_colors,
    }

    os.makedirs(DOCS, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nBuilt {OUTPUT}: {len(nodes)} nodes, {len(links)} links, {len(clusters_list)} communities")


if __name__ == "__main__":
    main()
