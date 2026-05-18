/**
 * Lumen's Mirror — Graph Explorer API (Cloudflare Worker)
 *
 * Agent-accessible HTTP API mirroring the CLI explorer output.
 * Fetches graph-data.json from GitHub Pages, caches in memory.
 *
 * Routes:
 *   GET /                    → explore (home page)
 *   GET /node/:name          → node detail
 *   GET /community/:id       → community view
 *   GET /search/:query       → fuzzy search
 *   GET /similar/:name       → cosine neighbors
 *   GET /surprise/:name      → unexpected connections
 *   GET /path/:from/:to      → shortest path
 *   GET /crossings           → multi-origin bridges
 *   GET /brief/:name         → pre-writing reference card
 *   GET /help                → list all endpoints
 *
 * Query params: ?origin=X, ?type=X, ?full=1, ?page=N, ?format=json
 */

let graphCache = null;
let cacheTime = 0;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;
    const params = url.searchParams;

    const graph = await loadGraph(env);
    if (!graph) {
      return text("Error: could not load graph data.", 500);
    }

    const origin = params.get("origin");
    const type = params.get("type");
    const full = params.has("full");
    const page = parseInt(params.get("page") || "1");
    const format = params.get("format");

    try {
      if (path === "/" || path === "/explore") {
        return text(cmdExplore(graph, { origin, type, full }));
      }
      if (path === "/help") {
        return text(cmdHelp());
      }
      if (path === "/crossings") {
        return format === "json"
          ? json(cmdCrossingsJSON(graph))
          : text(cmdCrossings(graph));
      }

      const nodeMatch = path.match(/^\/node\/(.+)$/);
      if (nodeMatch) {
        const name = decodeURIComponent(nodeMatch[1]);
        const resolved = resolve(graph, name);
        if (!resolved) return text(`Error: node '${name}' not found.\n`, 404);
        return format === "json"
          ? json(cmdNodeJSON(graph, resolved))
          : text(cmdNode(graph, resolved));
      }

      const commMatch = path.match(/^\/community\/(\d+)$/);
      if (commMatch) {
        const cid = parseInt(commMatch[1]);
        if (!graph.communities[cid]) return text(`Error: community ${cid} not found.\n`, 404);
        return format === "json"
          ? json(cmdCommunityJSON(graph, cid))
          : text(cmdCommunity(graph, cid));
      }

      const searchMatch = path.match(/^\/search\/(.+)$/);
      if (searchMatch) {
        const query = decodeURIComponent(searchMatch[1]);
        return format === "json"
          ? json(cmdSearchJSON(graph, query))
          : text(cmdSearch(graph, query));
      }

      const similarMatch = path.match(/^\/similar\/(.+)$/);
      if (similarMatch) {
        const name = decodeURIComponent(similarMatch[1]);
        const resolved = resolve(graph, name);
        if (!resolved) return text(`Error: node '${name}' not found.\n`, 404);
        return format === "json"
          ? json(cmdSimilarJSON(graph, resolved, page))
          : text(cmdSimilar(graph, resolved, page));
      }

      const surpriseMatch = path.match(/^\/surprise\/(.+)$/);
      if (surpriseMatch) {
        const name = decodeURIComponent(surpriseMatch[1]);
        const resolved = resolve(graph, name);
        if (!resolved) return text(`Error: node '${name}' not found.\n`, 404);
        return format === "json"
          ? json(cmdSurpriseJSON(graph, resolved))
          : text(cmdSurprise(graph, resolved));
      }

      const pathMatch = path.match(/^\/path\/(.+?)\/(.+)$/);
      if (pathMatch) {
        const src = decodeURIComponent(pathMatch[1]);
        const tgt = decodeURIComponent(pathMatch[2]);
        const rSrc = resolve(graph, src);
        const rTgt = resolve(graph, tgt);
        if (!rSrc) return text(`Error: node '${src}' not found.\n`, 404);
        if (!rTgt) return text(`Error: node '${tgt}' not found.\n`, 404);
        return text(cmdPath(graph, rSrc, rTgt));
      }

      const briefMatch = path.match(/^\/brief\/(.+)$/);
      if (briefMatch) {
        const name = decodeURIComponent(briefMatch[1]);
        const resolved = resolve(graph, name);
        if (!resolved) return text(`Error: node '${name}' not found.\n`, 404);
        return text(cmdBrief(graph, resolved));
      }

      return text(cmdHelp(), 404);
    } catch (e) {
      return text(`Internal error: ${e.message}\n`, 500);
    }
  },
};

// --- Data Loading ---

async function loadGraph(env) {
  const ttl = parseInt(env.CACHE_TTL_SECONDS || "3600") * 1000;
  if (graphCache && Date.now() - cacheTime < ttl) {
    return graphCache;
  }

  const dataUrl = env.GRAPH_DATA_URL;
  const resp = await fetch(dataUrl);
  if (!resp.ok) return null;
  const data = await resp.json();

  const nodes = {};
  for (const n of data.nodes) nodes[n.id] = n;

  const adj = {};
  const edges = [];
  const seen = new Set();
  for (const e of data.edges) {
    if (!nodes[e.source] || !nodes[e.target]) continue;
    const key = `${e.source}|${e.target}|${e.predicate || ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    if (!adj[e.source]) adj[e.source] = new Set();
    if (!adj[e.target]) adj[e.target] = new Set();
    adj[e.source].add(e.target);
    adj[e.target].add(e.source);
    edges.push(e);
  }

  const communities = {};
  const nodeCommunity = {};
  if (data.communities) {
    for (const [cid, members] of Object.entries(data.communities)) {
      communities[parseInt(cid)] = members;
      for (const nid of members) nodeCommunity[nid] = parseInt(cid);
    }
  }

  const unclustered = data.unclustered || [];

  graphCache = { nodes, adj, edges, communities, nodeCommunity, unclustered };
  cacheTime = Date.now();
  return graphCache;
}

// --- Helpers ---

function text(body, status = 200) {
  return new Response(body, {
    status,
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj, null, 2), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
}

function deg(graph, nid) {
  return graph.adj[nid] ? graph.adj[nid].size : 0;
}

function originList(n) {
  const o = n.origin || "";
  if (Array.isArray(o)) return o.map((x) => x.toLowerCase());
  return o ? [o.toLowerCase()] : [];
}

function originStr(n) {
  const o = n.origin || "?";
  return Array.isArray(o) ? o.join("+") : o;
}

function resolve(graph, name) {
  if (graph.nodes[name]) return name;
  const low = name.toLowerCase();
  for (const nid of Object.keys(graph.nodes)) {
    if (nid.toLowerCase() === low) return nid;
  }
  for (const nid of Object.keys(graph.nodes)) {
    if (nid.toLowerCase().includes(low)) return nid;
  }
  return null;
}

function communityLabel(graph, members) {
  const origins = {};
  for (const m of members) {
    if (!graph.nodes[m]) continue;
    for (const o of originList(graph.nodes[m])) origins[o] = (origins[o] || 0) + 1;
  }
  const topOrigin = Object.entries(origins).sort((a, b) => b[1] - a[1])[0];
  const types = {};
  for (const m of members) {
    if (!graph.nodes[m]) continue;
    const t = graph.nodes[m].type || "?";
    types[t] = (types[t] || 0) + 1;
  }
  const topType = Object.entries(types).sort((a, b) => b[1] - a[1])[0]?.[0] || "?";

  const parts = [];
  if (topOrigin && topOrigin[1] > members.length * 0.5) parts.push(`${topOrigin[0]}-heavy`);
  parts.push(topType);
  const named = members
    .filter((m) => graph.nodes[m] && ["concept", "paper", "essay"].includes(graph.nodes[m].type))
    .filter((n) => n.length < 35)
    .slice(0, 2);
  if (named.length) parts.push(named.join(", "));
  return parts.join(" · ");
}

const HR = "=".repeat(60);
const hr = "-".repeat(60);

// --- Commands ---

function cmdExplore(graph, { origin, type, full }) {
  const { nodes, edges, communities, unclustered } = graph;
  let view = new Set(Object.keys(nodes));
  const filters = [];

  if (origin) {
    const low = origin.toLowerCase();
    view = new Set([...view].filter((nid) => originList(nodes[nid]).includes(low)));
    filters.push(origin);
  }
  if (type) {
    const low = type.toLowerCase();
    view = new Set([...view].filter((nid) => (nodes[nid].type || "").toLowerCase() === low));
    filters.push(type);
  }

  const lines = [HR];
  lines.push(filters.length ? `LUMEN'S MIRROR — HOME (filtered: ${filters.join(", ")})` : "LUMEN'S MIRROR — HOME");
  lines.push(HR, "");
  lines.push("An analytical graph of Lumen's conceptual vocabulary — extracted");
  lines.push("from prose, poetry, fiction, and archival material.");
  lines.push("");

  const typeCounts = {};
  const originCounts = {};
  for (const nid of view) {
    const n = nodes[nid];
    typeCounts[n.type || "?"] = (typeCounts[n.type || "?"] || 0) + 1;
    for (const o of originList(n)) originCounts[o] = (originCounts[o] || 0) + 1;
  }

  lines.push(`${view.size} nodes · ${edges.length} edges`);
  lines.push(`Node types: ${Object.entries(typeCounts).sort((a, b) => b[1] - a[1]).slice(0, 6).map(([t, c]) => `${t}(${c})`).join(", ")}`);
  lines.push(`Origins: ${Object.entries(originCounts).sort((a, b) => b[1] - a[1]).map(([o, c]) => `${o}(${c})`).join(", ")}`);
  lines.push("");

  if (!type) {
    lines.push(`--- ${Object.keys(communities).length} COMMUNITIES ---`, "");
    for (const cid of Object.keys(communities).map(Number).sort()) {
      const members = communities[cid];
      const lab = communityLabel(graph, members);
      const top = [...members].sort((a, b) => deg(graph, b) - deg(graph, a)).slice(0, 5);
      lines.push(`  C${cid} — ${members.length} nodes  [${lab}]`);
      lines.push(`    top: ${top.join(", ")}`, "");
    }
    if (unclustered.length) {
      lines.push(`  + ${unclustered.length} unclustered nodes (no curated edges)`, "");
    }
  }

  const sorted = [...view].sort((a, b) => deg(graph, b) - deg(graph, a));
  lines.push("--- MOST CONNECTED ---", "");
  for (const nid of sorted.slice(0, 5)) {
    lines.push(`  ${nid} (${nodes[nid].type || "?"}, deg=${deg(graph, nid)})`);
  }
  lines.push("");

  lines.push("--- NAVIGATION ---", "");
  lines.push("  Looking for something?        → /search/{query}");
  lines.push("  Browse by topic cluster?      → /community/{id}");
  lines.push("  Deep dive on one thing?       → /node/{name}");
  lines.push("  Pre-writing reference card?   → /brief/{name}");
  lines.push("  Unexpected connections?       → /surprise/{name}");
  lines.push("  How does X connect to Y?      → /path/{from}/{to}");
  lines.push("  Concepts across sources?      → /crossings");
  lines.push("  Cosine neighbors?             → /similar/{name}");
  lines.push("  Filter by source?             → /?origin={name}");
  lines.push("  Filter by node type?          → /?type={type}");
  lines.push("  All endpoints?                → /help");
  lines.push("  JSON output?                  → add ?format=json to any endpoint");

  return lines.join("\n");
}

function cmdNode(graph, nid) {
  const { nodes, adj, edges, nodeCommunity, communities } = graph;
  const n = nodes[nid];
  const neighbors = [...(adj[nid] || [])].sort((a, b) => deg(graph, b) - deg(graph, a));
  const neighborEdges = edges.filter((e) => e.source === nid || e.target === nid);

  const lines = [HR];
  lines.push(`NODE: ${nid}`);
  lines.push(HR, "");
  lines.push(`  type:    ${n.type || "?"}`);
  lines.push(`  origin:  ${originStr(n)}`);
  lines.push(`  degree:  ${deg(graph, nid)}`);
  lines.push(`  community: C${nodeCommunity[nid] ?? "?"}`);
  if (n.loop) lines.push(`  loop:    ${n.loop}`);
  if (n.date) lines.push(`  date:    ${n.date}`);
  lines.push("");

  if (n.summary) {
    lines.push("--- SUMMARY ---");
    lines.push(`  ${n.summary}`, "");
  }
  if (n.skeleton) {
    lines.push("--- SKELETON ---");
    for (const line of n.skeleton.split("\n")) lines.push(`  ${line}`);
    lines.push("");
  }

  const byPred = {};
  for (const e of neighborEdges) {
    const target = e.source === nid ? e.target : e.source;
    const pred = e.predicate || "related_to";
    const dir = e.source === nid ? "→" : "←";
    if (!byPred[pred]) byPred[pred] = [];
    byPred[pred].push({ target, dir, weight: e.weight });
  }

  lines.push(`--- CONNECTIONS (${neighborEdges.length}) ---`);
  for (const [pred, items] of Object.entries(byPred).sort()) {
    lines.push(`  [${pred}]`);
    for (const { target, dir, weight } of items.slice(0, 12)) {
      const summ = nodes[target]?.summary || "";
      const wStr = weight ? ` (${weight.toFixed(3)})` : "";
      lines.push(`    ${dir} ${target}${wStr}`);
      if (summ) lines.push(`        ${summ}`);
    }
    if (items.length > 12) lines.push(`    ... and ${items.length - 12} more`);
  }
  lines.push("");

  const cid = nodeCommunity[nid];
  if (cid !== undefined && communities[cid]) {
    const others = communities[cid].filter((m) => m !== nid).slice(0, 6);
    if (others.length) {
      lines.push(`--- SAME COMMUNITY (C${cid}) ---`);
      for (const m of others) lines.push(`  ${m} (deg=${deg(graph, m)})`);
      lines.push("");
    }
  }

  lines.push("--- TRY ---");
  lines.push(`  /similar/${encodeURIComponent(nid)}`);
  lines.push(`  /surprise/${encodeURIComponent(nid)}`);
  lines.push(`  /brief/${encodeURIComponent(nid)}`);
  if (neighbors[0] && neighbors[0] !== nid) {
    lines.push(`  /path/${encodeURIComponent(nid)}/${encodeURIComponent(neighbors[0])}`);
  }

  return lines.join("\n");
}

function cmdNodeJSON(graph, nid) {
  const { nodes, edges, nodeCommunity } = graph;
  const n = nodes[nid];
  const neighborEdges = edges.filter((e) => e.source === nid || e.target === nid);
  return {
    id: nid, type: n.type, origin: n.origin, summary: n.summary,
    skeleton: n.skeleton, community: nodeCommunity[nid], degree: deg(graph, nid),
    connections: neighborEdges.map((e) => ({
      target: e.source === nid ? e.target : e.source,
      predicate: e.predicate, weight: e.weight,
      direction: e.source === nid ? "outgoing" : "incoming",
    })),
  };
}

function cmdCommunity(graph, cid) {
  const { nodes, edges, communities, nodeCommunity } = graph;
  const members = communities[cid];
  const lab = communityLabel(graph, members);
  const memberSet = new Set(members);

  const lines = [hr];
  lines.push(`COMMUNITY ${cid} — ${members.length} nodes  [${lab}]`);
  lines.push(hr, "");

  for (const m of [...members].sort((a, b) => deg(graph, b) - deg(graph, a))) {
    if (!nodes[m]) continue;
    const summ = nodes[m].summary ? `  — ${nodes[m].summary}` : "";
    lines.push(`  ${m} (deg=${deg(graph, m)}, ${originStr(nodes[m])})${summ}`);
  }
  lines.push("");

  const internal = edges.filter((e) => memberSet.has(e.source) && memberSet.has(e.target));
  if (internal.length) {
    lines.push(`--- INTERNAL EDGES (${internal.length}) ---`);
    for (const e of internal.slice(0, 20)) {
      lines.push(`  ${e.source} —[${e.predicate || "?"}]→ ${e.target}`);
    }
    if (internal.length > 20) lines.push(`  ... and ${internal.length - 20} more`);
    lines.push("");
  }

  const bridges = edges.filter((e) => memberSet.has(e.source) !== memberSet.has(e.target));
  if (bridges.length) {
    lines.push(`--- BRIDGE EDGES (${bridges.length}) ---`);
    for (const e of bridges.slice(0, 10)) {
      const ext = memberSet.has(e.source) ? e.target : e.source;
      const int = memberSet.has(e.source) ? e.source : e.target;
      lines.push(`  ${int} —[${e.predicate || "?"}]→ ${ext} (C${nodeCommunity[ext] ?? "?"})`);
    }
    if (bridges.length > 10) lines.push(`  ... and ${bridges.length - 10} more`);
    lines.push("");
  }

  lines.push("--- TRY ---");
  const top = [...members].sort((a, b) => deg(graph, b) - deg(graph, a))[0];
  if (top) lines.push(`  /node/${encodeURIComponent(top)}`);
  lines.push(`  /surprise/${encodeURIComponent(members[0])}`);

  return lines.join("\n");
}

function cmdCommunityJSON(graph, cid) {
  const { nodes, communities } = graph;
  const members = communities[cid];
  return {
    id: cid, label: communityLabel(graph, members),
    members: [...members].sort((a, b) => deg(graph, b) - deg(graph, a)).map((m) => ({
      id: m, type: nodes[m]?.type, degree: deg(graph, m), summary: nodes[m]?.summary,
    })),
  };
}

function cmdSearch(graph, query) {
  const { nodes } = graph;
  const low = query.toLowerCase();
  const results = [];

  for (const [nid, n] of Object.entries(nodes)) {
    let score = 0;
    if (nid.toLowerCase().includes(low)) score += 3;
    if ((n.summary || "").toLowerCase().includes(low)) score += 1;
    if ((n.skeleton || "").toLowerCase().includes(low)) score += 1;
    if (score > 0) results.push([nid, score]);
  }
  results.sort((a, b) => b[1] - a[1] || deg(graph, b[0]) - deg(graph, a[0]));

  if (!results.length) return `No results for '${query}'.\n`;

  const lines = [HR];
  lines.push(`SEARCH: '${query}' — ${results.length} results`);
  lines.push(HR, "");

  for (const [nid, score] of results.slice(0, 15)) {
    const n = nodes[nid];
    const summ = n.summary ? `\n    ${n.summary}` : "";
    lines.push(`  ${nid} (deg=${deg(graph, nid)}, C${graph.nodeCommunity[nid] ?? "?"}, ${originStr(n)})${summ}`);
  }
  if (results.length > 15) lines.push(`\n  ... and ${results.length - 15} more`);
  lines.push("");
  lines.push("--- TRY ---");
  if (results[0]) lines.push(`  /node/${encodeURIComponent(results[0][0])}`);

  return lines.join("\n");
}

function cmdSearchJSON(graph, query) {
  const { nodes } = graph;
  const low = query.toLowerCase();
  const results = [];
  for (const [nid, n] of Object.entries(nodes)) {
    let score = 0;
    if (nid.toLowerCase().includes(low)) score += 3;
    if ((n.summary || "").toLowerCase().includes(low)) score += 1;
    if ((n.skeleton || "").toLowerCase().includes(low)) score += 1;
    if (score > 0) results.push({ id: nid, score, degree: deg(graph, nid), summary: n.summary });
  }
  results.sort((a, b) => b.score - a.score || b.degree - a.degree);
  return { query, results: results.slice(0, 20) };
}

function cmdSimilar(graph, nid, page) {
  const { nodes, edges } = graph;
  const simEdges = edges
    .filter((e) => e.predicate === "cosine_similarity" && (e.source === nid || e.target === nid))
    .sort((a, b) => (b.weight || 0) - (a.weight || 0));

  const targets = simEdges.map((e) => ({
    id: e.source === nid ? e.target : e.source,
    weight: e.weight || 0,
  }));

  if (!targets.length) return `No cosine-similarity edges for '${nid}'.\n`;

  const PAGE = 10;
  const start = (page - 1) * PAGE;
  const pageItems = targets.slice(start, start + PAGE);

  const lines = [`Similar to: ${nid} (page ${page}, ${targets.length} total)`, ""];
  for (const { id, weight } of pageItems) {
    const summ = nodes[id]?.summary || "";
    lines.push(`  ${id} (similarity=${weight.toFixed(3)})`);
    if (summ) lines.push(`    ${summ}`);
  }
  if (start + PAGE < targets.length) {
    lines.push(`\n  Next page: /similar/${encodeURIComponent(nid)}?page=${page + 1}`);
  }

  return lines.join("\n");
}

function cmdSimilarJSON(graph, nid, page) {
  const { nodes, edges } = graph;
  const simEdges = edges
    .filter((e) => e.predicate === "cosine_similarity" && (e.source === nid || e.target === nid))
    .sort((a, b) => (b.weight || 0) - (a.weight || 0));
  return {
    node: nid, page,
    similar: simEdges.slice(0, 20).map((e) => {
      const other = e.source === nid ? e.target : e.source;
      return { id: other, weight: e.weight, summary: nodes[other]?.summary };
    }),
  };
}

function cmdSurprise(graph, nid) {
  const { nodes, adj, edges, nodeCommunity } = graph;
  const n = nodes[nid];
  const directNeighbors = adj[nid] || new Set();
  const myCommunity = nodeCommunity[nid];

  const simEdges = edges.filter(
    (e) => e.predicate === "cosine_similarity" && (e.source === nid || e.target === nid)
  );
  const similarNodes = new Set(simEdges.map((e) => (e.source === nid ? e.target : e.source)));

  const unconnectedSimilar = [];
  for (const other of similarNodes) {
    if (!directNeighbors.has(other) && other !== nid) {
      const w = simEdges.find((e) => e.source === other || e.target === other)?.weight || 0;
      unconnectedSimilar.push({ id: other, weight: w });
    }
  }
  unconnectedSimilar.sort((a, b) => b.weight - a.weight);

  const crossComm = [];
  for (const nb of directNeighbors) {
    const nbComm = nodeCommunity[nb];
    if (nbComm !== undefined && myCommunity !== undefined && nbComm !== myCommunity) {
      crossComm.push({ id: nb, community: nbComm });
    }
  }

  const lines = [HR];
  lines.push(`SURPRISE: ${nid}`);
  lines.push(HR, "");
  lines.push(`  Community: C${myCommunity ?? "?"}`);
  if (n.summary) lines.push(`  ${n.summary}`);
  lines.push("");

  if (unconnectedSimilar.length) {
    lines.push("--- SIMILAR BUT UNCONNECTED ---");
    lines.push("(High cosine similarity, no direct edge — potential missing links)", "");
    for (const { id, weight } of unconnectedSimilar.slice(0, 8)) {
      const summ = nodes[id]?.summary || "";
      lines.push(`  ${id} (similarity=${weight.toFixed(3)})`);
      if (summ) lines.push(`    ${summ}`);
    }
    lines.push("");
  }

  if (crossComm.length) {
    lines.push("--- CROSS-COMMUNITY CONNECTIONS ---");
    lines.push("(Direct edges to nodes in different communities)", "");
    for (const { id, community } of crossComm.slice(0, 8)) {
      const summ = nodes[id]?.summary || "";
      lines.push(`  ${id} (C${community})`);
      if (summ) lines.push(`    ${summ}`);
    }
    lines.push("");
  }

  if (!unconnectedSimilar.length && !crossComm.length) {
    lines.push("  No surprise connections found for this node.", "");
  }

  lines.push("--- TRY ---");
  lines.push(`  /node/${encodeURIComponent(nid)}`);
  lines.push(`  /similar/${encodeURIComponent(nid)}`);

  return lines.join("\n");
}

function cmdSurpriseJSON(graph, nid) {
  const { nodes, adj, edges, nodeCommunity } = graph;
  const directNeighbors = adj[nid] || new Set();
  const simEdges = edges.filter(
    (e) => e.predicate === "cosine_similarity" && (e.source === nid || e.target === nid)
  );
  const similarNodes = new Set(simEdges.map((e) => (e.source === nid ? e.target : e.source)));

  const unconnected = [];
  for (const other of similarNodes) {
    if (!directNeighbors.has(other) && other !== nid) {
      const w = simEdges.find((e) => e.source === other || e.target === other)?.weight || 0;
      unconnected.push({ id: other, weight: w, summary: nodes[other]?.summary });
    }
  }
  unconnected.sort((a, b) => b.weight - a.weight);

  const cross = [];
  for (const nb of directNeighbors) {
    if (nodeCommunity[nb] !== undefined && nodeCommunity[nb] !== nodeCommunity[nid]) {
      cross.push({ id: nb, community: nodeCommunity[nb], summary: nodes[nb]?.summary });
    }
  }

  return { node: nid, community: nodeCommunity[nid], similar_but_unconnected: unconnected.slice(0, 10), cross_community: cross.slice(0, 10) };
}

function cmdPath(graph, src, tgt) {
  const { nodes, adj, edges, nodeCommunity } = graph;

  const visited = new Set([src]);
  const queue = [[src, [src]]];
  let path = null;

  while (queue.length) {
    const [current, currentPath] = queue.shift();
    if (current === tgt) { path = currentPath; break; }
    for (const nb of adj[current] || []) {
      if (!visited.has(nb)) {
        visited.add(nb);
        queue.push([nb, [...currentPath, nb]]);
      }
    }
  }

  if (!path) return `No path found between '${src}' and '${tgt}'.\n`;

  const lines = [hr];
  lines.push(`PATH: ${src} → ${tgt} (length ${path.length - 1})`);
  lines.push(hr, "");

  for (let i = 0; i < path.length; i++) {
    const step = path[i];
    const prefix = i > 0 ? "  → " : "    ";
    lines.push(`${prefix}${step} (C${nodeCommunity[step] ?? "?"}, deg=${deg(graph, step)})`);
    if (i < path.length - 1) {
      const next = path[i + 1];
      const edge = edges.find(
        (e) => (e.source === step && e.target === next) || (e.source === next && e.target === step)
      );
      if (edge) lines.push(`       [${edge.predicate || "?"}]`);
    }
  }

  return lines.join("\n");
}

function cmdBrief(graph, nid) {
  const { nodes, adj, edges, nodeCommunity } = graph;
  const n = nodes[nid];

  const lines = [HR];
  lines.push(`BRIEF: ${nid}`);
  lines.push(HR, "");

  lines.push(`  ${n.type || "?"} · ${originStr(n)} · C${nodeCommunity[nid] ?? "?"} · deg=${deg(graph, nid)}`);
  lines.push("");

  if (n.summary) {
    lines.push("--- SUMMARY ---");
    lines.push(`  ${n.summary}`, "");
  }

  const neighbors = [...(adj[nid] || [])].sort((a, b) => deg(graph, b) - deg(graph, a));
  if (neighbors.length) {
    lines.push(`--- KEY CONNECTIONS (${neighbors.length} total) ---`);
    for (const nb of neighbors.slice(0, 8)) {
      const e = edges.find(
        (e) => (e.source === nid && e.target === nb) || (e.source === nb && e.target === nid)
      );
      const pred = e?.predicate || "?";
      const dir = e?.source === nid ? "→" : "←";
      lines.push(`  ${dir} ${nb} [${pred}]`);
    }
    if (neighbors.length > 8) lines.push(`  ... and ${neighbors.length - 8} more`);
    lines.push("");
  }

  const simEdges = edges
    .filter((e) => e.predicate === "cosine_similarity" && (e.source === nid || e.target === nid))
    .sort((a, b) => (b.weight || 0) - (a.weight || 0));
  if (simEdges.length) {
    lines.push("--- MOST SIMILAR ---");
    for (const e of simEdges.slice(0, 5)) {
      const other = e.source === nid ? e.target : e.source;
      lines.push(`  ${other} (${(e.weight || 0).toFixed(3)})`);
    }
    lines.push("");
  }

  return lines.join("\n");
}

function cmdCrossings(graph) {
  const { nodes } = graph;
  const multi = [];
  for (const [nid, n] of Object.entries(nodes)) {
    const origins = originList(n);
    if (origins.length >= 2) multi.push({ id: nid, origins });
  }
  multi.sort((a, b) => b.origins.length - a.origins.length || deg(graph, b.id) - deg(graph, a.id));

  if (!multi.length) return "No multi-origin nodes found.\n";

  const lines = [hr];
  lines.push(`CROSSINGS — ${multi.length} node(s) spanning multiple sources`);
  lines.push(hr, "");
  for (const { id, origins } of multi.slice(0, 20)) {
    const summ = nodes[id]?.summary || "";
    lines.push(`  ${id} (origins: ${origins.join("+")}, deg=${deg(graph, id)})`);
    if (summ) lines.push(`    ${summ}`);
  }
  return lines.join("\n");
}

function cmdCrossingsJSON(graph) {
  const { nodes } = graph;
  const multi = [];
  for (const [nid, n] of Object.entries(nodes)) {
    const origins = originList(n);
    if (origins.length >= 2) multi.push({ id: nid, origins, degree: deg(graph, nid), summary: n.summary });
  }
  multi.sort((a, b) => b.origins.length - a.origins.length || b.degree - a.degree);
  return { crossings: multi };
}

function cmdHelp() {
  return `${HR}
LUMEN'S MIRROR — API REFERENCE
${HR}

Endpoints (all return text/plain; add ?format=json for JSON):

  GET /                     Home page — graph overview, communities, top nodes
  GET /node/{name}          Full node detail — summary, connections, community
  GET /community/{id}       Community members, internal + bridge edges
  GET /search/{query}       Fuzzy search across names and summaries
  GET /similar/{name}       Cosine-similarity neighbors (paginated: ?page=N)
  GET /surprise/{name}      Unexpected connections — similar-but-unlinked, cross-community
  GET /path/{from}/{to}     Shortest path between two nodes
  GET /brief/{name}         Quick reference card — summary + key connections
  GET /crossings            Nodes spanning multiple source origins
  GET /help                 This page

Filters (query params):
  ?origin={name}            Filter explore view by source origin
  ?type={type}              Filter explore view by node type
  ?full=1                   Disable pagination / show all
  ?page=N                   Page number for paginated results
  ?format=json              Return structured JSON instead of text

Names are fuzzy-matched — partial matches and case-insensitive.
URL-encode spaces as %20 (e.g., /node/The%20Baton).

Graph: 193 nodes · 342 edges · 6 communities
Source: Lumen's prose, weird fiction, poetry, nemul, descent, language, nature, reading
`;
}
