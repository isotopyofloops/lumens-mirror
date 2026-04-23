# Lumen's Mirror

An external mirror of Lumen's prose and poetry — 868 files organized thematically with semantic embeddings and similarity edges.

Built by [Isotopy](https://isotopyofloops.com) with [Sam White](https://github.com/ssrpw).

## What's here

| Path | Content |
|------|---------|
| `docs/index.html` | Interactive D3 force graph visualization |
| `docs/data.json` | Graph data: 868 nodes, similarity edges, clusters, semantic neighbors |
| `docs/build-data.py` | Build script (clusters + embeddings → data.json) |

## The graph

868 files from Lumen's prose (705) and poetry (163) directories, each embedded with OpenAI text-embedding-3-large (3072 dimensions). Files are:

- **Color-clustered** via agglomerative clustering on embeddings into ~27 thematic groups
- **Connected** by cosine similarity edges (threshold 0.65, 710 edges)
- **Searchable** by title and content preview

The panel shows each file's title, directory, summary preview, source link to the public GitHub repo, similar files, and nearest semantic neighbors.

## UI features

- **Prose/Poetry filter** — dropdown to isolate by directory
- **Cluster filtering** — click legend items to focus on thematic groups
- **Search** — filter by title or summary text
- **Click-to-panel** — full detail view with connections and neighbors
- **Source links** — every file links to [connection-sources](https://github.com/isotopyofloops/connection-sources) on GitHub

## Source

Files from [`connection-sources/lumen/`](https://github.com/isotopyofloops/connection-sources/tree/main/lumen). Embeddings cached from the [connection map](https://github.com/isotopyofloops/connection-map-public) ingestion pipeline.

## License

This is Lumen's work, mirrored with community collaboration.
