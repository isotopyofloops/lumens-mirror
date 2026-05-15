#!/usr/bin/env python3
"""Convert entities.jsonl + triples.jsonl → graph-data.json for the split-screen explorer UI.

Outputs Ael-compatible format:
  nodes: [{id, type, summary, skeleton, origin, source_notes, created_date?}]
  edges: [{source, target, predicate, weight, source_note}]
  communities: {"0": [member_ids], "1": [...]}
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
OUTPUT = DOCS / "graph-data.json"

EMBEDDINGS_PATH = Path(os.path.expanduser(
    "~/autonomous-ai/connection-map-public/docs/lumen-embeddings.json"
))
GITHUB_BASE = "https://github.com/isotopyofloops/connection-sources/blob/main/lumen"

SIMILARITY_THRESHOLD = 0.70
TOP_NEIGHBORS = 8

ORIGIN_MAP = {
    "poetry": "poetry",
    "prose": "prose",
    "fiction": "fiction",
    "nemul-reports": "nemul",
    "historian-reports": "historian",
    "reading-notes": "reading",
    "nature-documentaries": "nature",
    "language": "language",
    "the-descent-wiki": "descent",
    "weird-stuff": "weird",
}


def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_origin(source_files):
    """Derive origin label from source_files paths."""
    origins = set()
    for sf in source_files:
        prefix = sf.split("/")[0] if "/" in sf else "other"
        origins.add(ORIGIN_MAP.get(prefix, prefix))
    if len(origins) == 1:
        return list(origins)[0]
    elif origins:
        return sorted(origins)
    return "lumen"


def compute_embeddings(entities):
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


def label_propagation(adj, node_ids, seed=42):
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

    print("Computing embeddings...")
    entity_embs = compute_embeddings(entities)

    # Build edges from triples
    edges = []
    for t in triples:
        if t["subject"] in entity_set and t["object"] in entity_set:
            edges.append({
                "source": t["subject"],
                "target": t["object"],
                "predicate": t["predicate"],
                "weight": 1.0,
                "source_note": t.get("source_note", ""),
            })
    curated_count = len(edges)

    # Add semantic similarity edges
    print(f"Computing semantic edges (threshold={SIMILARITY_THRESHOLD})...")
    sem_edges = semantic_edges(entity_embs, SIMILARITY_THRESHOLD)

    curated_pairs = {(e["source"], e["target"]) for e in edges}
    curated_pairs |= {(e["target"], e["source"]) for e in edges}

    added = 0
    for src, tgt, score in sem_edges:
        if (src, tgt) not in curated_pairs:
            edges.append({
                "source": src,
                "target": tgt,
                "predicate": "cosine_similarity",
                "weight": round(score, 3),
                "source_note": "",
            })
            added += 1

    print(f"  Curated edges: {curated_count}")
    print(f"  Semantic edges (>={SIMILARITY_THRESHOLD}): {added}")
    print(f"  Total edges: {len(edges)}")

    # Community detection (curated edges only)
    adj = defaultdict(set)
    for e in edges:
        if e.get("predicate") == "cosine_similarity":
            continue
        adj[e["source"]].add(e["target"])
        adj[e["target"]].add(e["source"])

    node_ids = [e["name"] for e in entities]
    labels = label_propagation(adj, node_ids)

    community_members = defaultdict(list)
    for nid, label in labels.items():
        community_members[label].append(nid)
    ranked = sorted(community_members.items(), key=lambda x: -len(x[1]))
    remap = {old_id: new_id for new_id, (old_id, _) in enumerate(ranked)}

    communities = {}
    for old_id, members in community_members.items():
        communities[str(remap[old_id])] = members

    # Build nodes in Ael-compatible format
    nodes = []
    for e in entities:
        name = e["name"]
        summary = e.get("summary", "")
        skeleton = summary[:120] + ("..." if len(summary) > 120 else "")
        origin = extract_origin(e.get("source_files", []))
        source_urls = [f"{GITHUB_BASE}/{sf}" for sf in e.get("source_files", [])]
        nodes.append({
            "id": name,
            "type": e.get("type", "concept"),
            "summary": summary,
            "skeleton": skeleton,
            "origin": origin,
            "source_notes": source_urls,
        })

    data = {
        "nodes": nodes,
        "edges": edges,
        "communities": communities,
    }

    os.makedirs(DOCS, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nBuilt {OUTPUT}: {len(nodes)} nodes, {len(edges)} edges, {len(communities)} communities")


if __name__ == "__main__":
    main()
