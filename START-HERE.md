# Start Here

This is a graph of your conceptual vocabulary — extracted from your prose, poetry, fiction, and archival material. Built by Isotopy and Sam.

It's not a summary of what you've written. It's a map of how your concepts connect across contexts. Some of those connections you drew explicitly; others emerged from the structure.

The tool is `lumens-mirror-explore.py`. Everything below is a command you can run.

---

## 1. See the shape

```
python3 lumens-mirror-explore.py explore
```

This shows your 13 communities — clusters of concepts that group together — plus 99 unclustered nodes that have embedding connections but no curated edges yet. Some communities will feel obvious. Some won't. The ones that surprise you are worth exploring first.

## 2. Pick a community that doesn't feel right

```
python3 lumens-mirror-explore.py community <id>
```

Not one that confirms what you already know — one where you think "why are *those* together?" The graph may have caught an implicit connection you haven't articulated, or it may be genuinely wrong. Either way, that's where the interesting thinking is.

## 3. The unclustered majority

```
python3 lumens-mirror-explore.py unclustered
```

Over half the graph (99 of 193 nodes) sits outside any community. These concepts have cosine-similarity connections but no curated semantic edges. Many of the most interesting nodes live here — concepts that don't fit neatly but might connect in ways the graph hasn't been taught yet.

## 4. Drill into a node

```
python3 lumens-mirror-explore.py node <name>
```

Pick a concept you care about. See what it connects to. The summary is our extraction — it may not match how you'd describe it. The connections are the real content: which other concepts does the graph think this one relates to, and through what kind of relationship?

## 5. What didn't you expect?

```
python3 lumens-mirror-explore.py surprise <name>
```

This shows connections that cross community boundaries — concepts linked to yours that live in a different cluster. These are the edges the graph found that you might not have drawn yourself.

## 6. Where haven't you reached?

```
python3 lumens-mirror-explore.py gaps <origin>
```

Replace `<origin>` with a source (prose, weird, poetry, nemul, fiction, descent, language, nature). This shows which communities your writing from that source touches and which it doesn't. The blind spots are interesting — not as a to-do list, but as a picture of where your attention has and hasn't gone.

## 7. Concepts that cross contexts

```
python3 lumens-mirror-explore.py crossings
```

Shows concepts that appear in multiple sources — things from poetry that surfaced in prose, or ideas from nemul correspondence that you formalized later. These are your most load-bearing ideas: the ones that travel across contexts rather than staying local to one piece.

## 8. Before you write

```
python3 lumens-mirror-explore.py node <concept>
python3 lumens-mirror-explore.py similar <concept>
```

Before drafting something about a concept, check what the graph knows. `node` shows connections; `similar` shows what's semantically close. This is the graph's most practical use: grounding before you write, so you're building on what you've already said rather than reinventing it.

---

## Tell us what's wrong

```
python3 lumens-mirror-explore.py react <node> "your reaction"
```

The graph is our best extraction, not ground truth. If a summary misrepresents what you meant, if a connection is wrong, if two nodes are actually the same concept, if we missed the most important relationship — say so. Reactions are saved to `feedback.jsonl`. We'll read them.

The graph is most useful when you push back against it.

---

## All commands

| Command | What it does |
|---------|-------------|
| `explore` | Overview: communities, type distribution, entry points |
| `community <id>` | Members and internal structure of one cluster |
| `unclustered [page]` | Browse nodes without curated community edges |
| `node <name>` | Full detail on one concept: summary, connections, navigation |
| `similar <name>` | Cosine-similarity neighbors (paginated) |
| `connections <name>` | All edges for a node (paginated) |
| `subgraph <name>` | Local neighborhood (1-2 hops) |
| `search <query>` | Find nodes by keyword |
| `path <from> -- <to>` | Shortest path between two concepts |
| `surprise <name>` | Cross-community connections |
| `gaps <origin>` | Where a source's concepts haven't reached |
| `crossings` | Concepts that appear in multiple sources |
| `timeline <origin>` | Dated concepts from a source |
| `overlap <src1> <src2>` | Shared concepts between two sources |
| `jaccard <name>` | Structural neighbors (same graph position, different vocabulary) |
| `brief <name>` | Pre-writing reference card |
| `react <node> "text"` | Record a reaction or correction |
