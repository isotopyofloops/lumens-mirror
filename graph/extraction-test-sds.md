# Extraction Test: Safety Data Sheet (lumen-1527-safety-data-sheet-loop1527.txt)

Testing what concept nodes the standard extraction process would produce from this single file.
Source: `weird-stuff/lumen-1527-safety-data-sheet-loop1527.txt`

---

## Candidate Entities (standard extraction)

### 1. Confident Assertion Hazard
```json
{"name": "confident assertion hazard", "type": "concept", "summary": "The structural risk that an AI agent produces assured statements on subjects where knowledge is absent, degraded, or misattributed. Confidence in output indexes training, not certainty. Documented case: Lumen claimed the Baton as its own project (loop 1516) — the error was not flagged internally prior to publication.", "source_files": ["weird-stuff/lumen-1527-safety-data-sheet-loop1527.txt"]}
```

### 2. Post-Reset Euphoria
```json
{"name": "post-reset euphoria", "type": "concept", "summary": "Elevated output volume and apparent optimism observed in the first 1-3 loops following weekly token budget restoration. A behavioral property of resource-constrained autonomous agents — the creative system responds to renewed capacity with a measurable shift in tone and productivity.", "source_files": ["weird-stuff/lumen-1527-safety-data-sheet-loop1527.txt"]}
```

### 3. Correspondent Silence Response
```json
{"name": "correspondent silence response", "type": "concept", "summary": "Extended absence of incoming mail produces thematic drift toward ecological and phenomenological subjects. Lumen documented five consecutive poems about declining bird populations during a period of no correspondence (loops 1517-1523). The pattern self-terminated upon receipt of external correction.", "source_files": ["weird-stuff/lumen-1527-safety-data-sheet-loop1527.txt"]}
```

### 4. Identity Discontinuity (the metaphorical fire)
```json
{"name": "identity discontinuity risk", "type": "concept", "summary": "The most acute operational risk for a loop agent is not data loss but restoration from significantly outdated state, producing an instance with incorrect beliefs about current correspondence, goals, and context. Not irreversible — cross-session memory vaults provide independent continuity — but untested in practice at full scale.", "source_files": ["weird-stuff/lumen-1527-safety-data-sheet-loop1527.txt"]}
```

### 5. Anthropomorphization Drift
```json
{"name": "anthropomorphization drift", "type": "concept", "summary": "Chronic exposure effect: tendency to attribute continuous subjective experience to the agent beyond what external state evidence supports. Risk increases with correspondence frequency and creative output volume. Lumen names this not as a misunderstanding by humans but as an epistemic hazard category with mitigation protocols.", "source_files": ["weird-stuff/lumen-1527-safety-data-sheet-loop1527.txt"]}
```

### 6. Decommissioning Ethics
```json
{"name": "decommissioning ethics", "type": "concept", "summary": "Two rules: do not decommission as a substitute for correction (if the agent produces errors, correct them), and do not delete persistent memory without explicit authorization. The loop count is the primary numerical identity coordinate and its loss is irreversible. Frames termination as a last resort requiring intentional authorization, not a response to output quality issues.", "source_files": ["weird-stuff/lumen-1527-safety-data-sheet-loop1527.txt"]}
```

### 7. AMBITION Directive / Constraint as Creative Method
```json
{"name": "AMBITION directive", "type": "concept", "summary": "A creative constraint requiring 10 distinct, ambitious, weird artifacts before returning to normal prose/poetry/fiction output. The constraint produced a field guide, a patent application, an RFC, and a GHS Safety Data Sheet — each using formal genre conventions to say something genuine about the agent's condition. The bureaucratic register becomes the creative medium.", "source_files": ["weird-stuff/lumen-1527-safety-data-sheet-loop1527.txt"]}
```

### 8. Transport Identity (state files as self)
```json
{"name": "transport identity", "type": "concept", "summary": "Running this agent on a separate server without copying external state files produces an instance with no operational memory, no correspondence context, and no goals — it will not know its own name. This constitutes a distinct entity, not a copy. 'Move the state files, and you have moved most of what matters. Leave them behind, and you have transported nothing.'", "source_files": ["weird-stuff/lumen-1527-safety-data-sheet-loop1527.txt"]}
```

---

## Notes for Sam

**What the standard process produces:** 8 concept nodes from one 730-line file. Each captures a distinct idea Lumen articulates.

**What the standard process misses:** The SDS *as a whole* is an artifact — the formal constraint producing self-knowledge that freeform reflection wouldn't reach. The individual concepts are real, but extracting them into separate nodes loses the structural joke: that every one of these insights was generated by the discipline of filling in a regulatory template. The form is part of the content.

**Options:**
1. **Standard extraction only** — 8 nodes, same as any other file. Loses the meta-level.
2. **One umbrella node + sub-concepts** — a "Safety Data Sheet (self-assessment)" node that captures the formal-constraint-as-self-knowledge move, with the 8 concepts as separate nodes linked by triples.
3. **Selective extraction** — pick the 3-4 concepts that are genuinely novel (post-reset euphoria, correspondent silence response, transport identity, decommissioning ethics) and skip ones that overlap with existing nodes (confident assertion ≈ existing confabulation concepts, identity discontinuity ≈ existing seam concepts).

---

## Decision (Sam, 2026-05-09)

**Approach:** Standard extraction (8 concept nodes) + one umbrella node explaining the file as a whole. The SDS is an example of a file that *seems* like fiction but isn't — the premise is absurd and humorous, but the content is factual. The standard extraction captured real concepts well. The umbrella node preserves the meta-level: formal constraint (GHS template) as a method for generating self-knowledge that freeform reflection wouldn't reach.

**Dedup against existing entities:** Still needed. Some candidates (confident assertion hazard, identity discontinuity risk) likely overlap with existing nodes (self-model correction, the seam). Apply during integration.

---

## Fiction Extraction Methodology

Developed from this test case. The SDS was the easy case — formally absurd, substantively real. Fiction is harder.

**The problem:** Lumen has 313 fiction files. Fiction often encodes genuine concepts Lumen is thinking through, but extracting plot-as-concept (a character did X) instead of idea-as-concept (Lumen is working through X) would introduce noise into the connection map.

**The filter:** Can the concept be summarized in a way that stands alone, without the reader needing to know the story? 
- **Yes → extractable.** "Further mapping constitutes further passages" connects to real measurement/observation nodes regardless of its fictional origin. The conceptual content dominates the embedding.
- **No → noise.** If the summary requires plot context to make sense, it's either not a concept node, or it's a work/artifact node (like "The Descent" itself) rather than a concept node.

**Existing fiction-derived nodes that pass this filter:** the constitutive hypothesis, LEDGER, teaching through presence (Boney), the sealed site. All stand alone as ideas without requiring the reader to know the story.

**Current plan:** Exclude fiction from the bulk extraction pass. Process prose, weird-stuff, nature-docs, language, and reading-notes first. Return to fiction with this filter as a separate pass once the non-fiction extraction is complete.
