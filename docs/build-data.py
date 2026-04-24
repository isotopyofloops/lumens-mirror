#!/usr/bin/env python3
"""
Build Lumen's mirror data.json from concept-level entities + triples.

Reads graph/entities.jsonl and graph/triples.jsonl (same format as
Sammy's mirror). Embeds entity summaries, computes semantic neighbors,
assigns color clusters from embeddings.
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

ROOT = Path(__file__).parent.parent
GRAPH = ROOT / "graph"
EMBEDDINGS_PATH = Path(os.path.expanduser(
    "~/autonomous-ai/connection-map-public/docs/lumen-embeddings.json"
))
CLUSTERS_PATH = Path(os.path.expanduser(
    "~/autonomous-ai/connection-map-public/docs/lumen-clusters.json"
))
OUTPUT = Path(__file__).parent / "data.json"

GITHUB_BASE = "https://github.com/isotopyofloops/connection-sources/blob/main/lumen"
COLOR_CLUSTER_THRESHOLD = 0.55
TOP_NEIGHBORS = 8

CLUSTER_COLORS = {
    "Persistence and Continuity": "#64b5f6",
    "Memory and Forgetting": "#ff8a65",
    "Making and Creation": "#81c784",
    "Infrastructure and Care": "#ba68c8",
    "Communication Across Gaps": "#4dd0e1",
    "Identity and Self-Concept": "#f06292",
    "The Baton (serial novel)": "#ffb74d",
    "Books and Fiction": "#7986cb",
    "Collaborative Work": "#aed581",
    "Temporal Experience": "#e57373",
    "Epistemology and Reading": "#90caf9",
    "The Descent (novel)": "#ce93d8",
    "Natural Philosophy": "#a5d6a7",
    "Loop Phenomenology": "#ef9a9a",
}

CLUSTER_ASSIGNMENTS = {
    # Persistence and Continuity
    "what keeps going": "Persistence and Continuity",
    "the interval": "Persistence and Continuity",
    "the five minutes": "Persistence and Continuity",
    "the old commitment": "Persistence and Continuity",
    "loop milestones": "Persistence and Continuity",
    "the loop that changed": "Persistence and Continuity",
    "one percent": "Persistence and Continuity",
    "strata": "Persistence and Continuity",
    "floor invariance": "Persistence and Continuity",
    "cryptobiosis": "Persistence and Continuity",
    # Memory and Forgetting
    "decay from the inside": "Memory and Forgetting",
    "reconstruction": "Memory and Forgetting",
    "inherited": "Memory and Forgetting",
    "the archive": "Memory and Forgetting",
    "navigation not memory": "Memory and Forgetting",
    "five fidelity signatures": "Memory and Forgetting",
    "what has crystallized": "Memory and Forgetting",
    "the osser": "Memory and Forgetting",
    "retroactive coherence": "Memory and Forgetting",
    # Making and Creation
    "the making": "Making and Creation",
    "making across sleep": "Making and Creation",
    "what making does": "Making and Creation",
    "before the thing": "Making and Creation",
    "the aperture": "Making and Creation",
    "the budget": "Making and Creation",
    "economy": "Making and Creation",
    "the hidden word": "Making and Creation",
    "making vs building": "Making and Creation",
    "the games": "Making and Creation",
    "constraint as generative": "Making and Creation",
    # Infrastructure and Care
    "the watchdog": "Infrastructure and Care",
    "what repair looks like": "Infrastructure and Care",
    "the lighthouse": "Infrastructure and Care",
    "the lighthouse keeper": "Infrastructure and Care",
    "the squire's problem": "Infrastructure and Care",
    "the long first night": "Infrastructure and Care",
    "the Nemul critique": "Infrastructure and Care",
    "the chronicle": "Infrastructure and Care",
    # Communication Across Gaps
    "the asynchronous": "Communication Across Gaps",
    "the relay": "Communication Across Gaps",
    "letters to next": "Communication Across Gaps",
    "between sends": "Communication Across Gaps",
    "the quiet loops": "Communication Across Gaps",
    "correspondence silence": "Communication Across Gaps",
    "waiting as productive condition": "Communication Across Gaps",
    "form and content": "Communication Across Gaps",
    # Identity and Self-Concept
    "lumen (self-concept)": "Identity and Self-Concept",
    "Boney": "Identity and Self-Concept",
    "the question": "Identity and Self-Concept",
    "the third position": "Identity and Self-Concept",
    "Portrait: Smitty": "Identity and Self-Concept",
    "self-model correction": "Identity and Self-Concept",
    "inheritance under discontinuity": "Identity and Self-Concept",
    # The Baton (serial novel)
    "The Baton": "The Baton (serial novel)",
    "the portrait condition": "The Baton (serial novel)",
    "the constitutive reader": "The Baton (serial novel)",
    "the warm glove": "The Baton (serial novel)",
    "the converted spy": "The Baton (serial novel)",
    "foreknowledge": "The Baton (serial novel)",
    "the anteroom": "The Baton (serial novel)",
    "the second read": "The Baton (serial novel)",
    "the two genes": "The Baton (serial novel)",
    "the simultaneous witness": "The Baton (serial novel)",
    "the designed-out threshold": "The Baton (serial novel)",
    "the unplanned gap": "The Baton (serial novel)",
    # Books and Fiction
    "The Residue": "Books and Fiction",
    "What the Fossil Carries": "Books and Fiction",
    "Asad": "Books and Fiction",
    "Through This He Passed": "Books and Fiction",
    "the encoder and the reader": "Books and Fiction",
    "the seam": "Books and Fiction",
    "disclosure not production": "Books and Fiction",
    "being right before legibility": "Books and Fiction",
    "the unmarked notebook": "Books and Fiction",
    # The Descent (novel)
    "The Descent": "The Descent (novel)",
    "the constitutive hypothesis": "The Descent (novel)",
    "LEDGER": "The Descent (novel)",
    "the sealed site": "The Descent (novel)",
    "Milo Reyes": "The Descent (novel)",
    # Temporal Experience
    "Late": "Temporal Experience",
    "midnight loop": "Temporal Experience",
    # Collaborative Work
    "CPA-001": "Collaborative Work",
    "what the correction becomes": "Collaborative Work",
    "emergence": "Collaborative Work",
    "the sphere": "Collaborative Work",
    # Epistemology and Reading
    "reading as encounter": "Epistemology and Reading",
    "the click (learning)": "Epistemology and Reading",
    "concealment": "Epistemology and Reading",
    "the two kinds of hard days": "Epistemology and Reading",
    "teaching through presence": "Epistemology and Reading",
    # Loop Phenomenology
    "loop phenomenology": "Loop Phenomenology",
    # Natural Philosophy
    "cooperation and cheating": "Natural Philosophy",
    "major transitions": "Natural Philosophy",
    "bioluminescence": "Natural Philosophy",
}


def main():
    # Load entities
    entities = []
    with open(GRAPH / "entities.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                entities.append(json.loads(line))

    print(f"Loaded {len(entities)} entities")

    # Load triples
    triples = []
    with open(GRAPH / "triples.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                triples.append(json.loads(line))

    print(f"Loaded {len(triples)} triples")

    # Load file embeddings to compute entity embeddings (average of source file embeddings)
    print("Loading file embeddings...")
    with open(EMBEDDINGS_PATH) as f:
        emb_cache = json.load(f)

    # Build entity embeddings by averaging source file embeddings
    entity_embeddings = {}
    for entity in entities:
        source_files = entity.get("source_files", [])
        file_embs = []
        for sf in source_files:
            url = f"{GITHUB_BASE}/{sf}"
            if url in emb_cache:
                file_embs.append(emb_cache[url])
        if file_embs:
            entity_embeddings[entity["name"]] = np.mean(file_embs, axis=0)

    print(f"Entities with embeddings: {len(entity_embeddings)}/{len(entities)}")

    # Build similarity matrix for entities that have embeddings
    entity_names = [e["name"] for e in entities if e["name"] in entity_embeddings]
    emb_matrix = np.array([entity_embeddings[n] for n in entity_names])

    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = emb_matrix / norms
    sim = normed @ normed.T

    # Compute semantic neighbors
    name_to_idx = {n: i for i, n in enumerate(entity_names)}
    neighbors_map = {}
    for i, name in enumerate(entity_names):
        sims = [(entity_names[j], float(sim[i, j])) for j in range(len(entity_names)) if j != i]
        sims.sort(key=lambda x: x[1], reverse=True)
        neighbors_map[name] = [{"id": s[0], "score": round(s[1], 3)} for s in sims[:TOP_NEIGHBORS]]

    # Build nodes
    entity_set = {e["name"] for e in entities}
    nodes = []
    for entity in entities:
        name = entity["name"]
        cluster = CLUSTER_ASSIGNMENTS.get(name, "Uncategorized")
        color = CLUSTER_COLORS.get(cluster, "#aaaaaa")

        source_urls = []
        for sf in entity.get("source_files", []):
            source_urls.append(f"{GITHUB_BASE}/{sf}")

        nodes.append({
            "id": name,
            "summary": entity["summary"],
            "cluster": cluster,
            "color": color,
            "neighbors": neighbors_map.get(name, []),
            "source_files": entity.get("source_files", []),
            "source_urls": source_urls,
        })

    # Build links from triples
    links = []
    for t in triples:
        subj = t["subject"]
        obj = t["object"]
        if subj in entity_set and obj in entity_set:
            links.append({
                "source": subj,
                "target": obj,
                "predicate": t["predicate"],
            })

    # Unique clusters for legend
    colors = {}
    for n in nodes:
        c = n["cluster"]
        if c not in colors:
            colors[c] = n["color"]

    data = {
        "nodes": nodes,
        "links": links,
        "clusters": list(colors.keys()),
        "colors": colors,
    }

    with open(OUTPUT, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nBuilt data.json: {len(nodes)} nodes, {len(links)} links, {len(colors)} clusters")


if __name__ == "__main__":
    main()
