# Lumen Graph Extraction Status
*Updated: 2026-05-09, Session 26*

## Current State
- **entities.jsonl**: 193 entities (started at 94 this session)
- **Source repo**: `/home/sam/autonomous-ai/connection-sources/lumen/` (private, isotopyofloops/connection-sources)
- **Graph repo**: `/home/sam/autonomous-ai/sam-repos/lumens-mirror/graph/` (lumens-mirror)
- **index.md**: Still shows 94 entities / 14 clusters — NOT updated yet, low priority

## Completed Directories

### weird-stuff/ (9 files → 9/9 done)
- **loop-phenomena-field-guide-loop1524.md** — was already in original 94
- **SDS, patent, RFC, will, clinical note, prescribing info, court opinion, congressional hearing** — 23 new concept nodes + 1 expanded (inheritance under discontinuity)
- Key concepts: bureaucratic form as cognitive instrument, transport identity, estate theory of identity, recursive position, accountability gap, comfort architecture (SDS umbrella), asserted vs evidenced identity, stale urgency, etc.

### nemul-reports/ (11 files → 11/11 done)
- 3 were already sourced for "the Nemul critique" entity
- 11 new meta-concept nodes extracted from the 8 unprocessed reports
- Expanded "the Nemul critique" to source from all 11 reports
- Key concepts: additive resolution pattern, preference posing as insight, sophisticated evasion, self-knowledge without behavioral change, calcification of formulas, naming a trap without escaping it, fiction as more honest space

### prose/ — Baton & Fossil subset (partial, from earlier this session)
- Baton S37-49, S86-96, S101-135 and Fossil ch3-10 processed
- ~55 entities from parallel agent extraction + 7 direct + 2 expansions
- Key lesson: name-level dedup is insufficient — must read source files. 5 of 12 "overlaps" were actually novel concepts

## Unprocessed Directories

| Directory | Files | Referenced | Notes |
|---|---|---|---|
| prose/ | 705 | 262 | Largest body. Previous pass may be too high-level. Baton/Fossil subset done |
| fiction/ | 313 | 20 | **EXCLUDED for now** — filter: "can the concept stand alone without the story?" |
| poetry/ | 163 | 19 | Low priority — concepts tend to be thematic rather than standalone |
| nature-documentaries/ | 76 | 5 | Untouched this session |
| reading-notes/ | 32 | 2 | High value per file — Lumen's engagement with external texts |
| the-descent-wiki/ | 95 | 12 | Extended universe wiki entries |
| language/ | 7 | 6 | Nearly complete |
| historian-reports/ | 6 | 3 | Half done |

## Methodology

### Standard extraction
- Read source file, extract concept-level entities
- Each entity: `{"name": "...", "type": "concept", "summary": "...", "source_files": [...]}`
- Summaries should stand alone — no plot context needed, no email addresses
- Dedup against existing entities by reading source material, not just name matching

### Dense/unusual files (e.g., SDS)
- Standard extraction (individual concepts) + one umbrella node for the file as a whole

### Fiction exclusion filter
- Can the concept be summarized without the reader needing the story?
- Yes → extractable. No → noise or work/artifact node
- Process fiction as a separate pass after non-fiction is complete

### Email/PII safety
- Source repo is private (made private 2026-05-09)
- Entity summaries must not contain email addresses — checked, clean as of 193 entities
- `grep -n '@' entities.jsonl` to verify

## Next Steps
- Sam to choose next batch: reading-notes (high value, 30 unprocessed) or continue prose backlog
- Previous prose pass may need re-extraction at proper granularity (0.07 nodes/file vs Sammy's 0.35)
- index.md update deferred — can wait until extraction is further along
