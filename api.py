#!/usr/bin/env python3
"""
Lumen's Mirror — Graph Explorer API

Thin HTTP wrapper around lumens-mirror-explore.py. Returns the exact same
output an agent would see running the CLI locally — breadcrumbs, pagination,
navigation hints, all of it.

Run:
    pip install fastapi uvicorn
    uvicorn api:app --port 8100

Usage:
    GET /                         → runs 'explore' (home page)
    GET /cmd/explore              → same as above
    GET /cmd/node/The%20Baton     → node detail
    GET /cmd/community/0          → community view
    GET /cmd/search/interval      → search
    GET /cmd/surprise/The%20Baton → unexpected connections
    GET /cmd/similar/the%20seam   → cosine neighbors
    GET /cmd/path/X/--/Y          → shortest path (use -- as separator)
    GET /cmd/brief/The%20Baton    → pre-writing reference card
    GET /cmd/crossings            → multi-origin concepts
    GET /cmd/timeline/prose       → source timeline
    GET /cmd/overlap/prose/weird  → compare two sources

    Flags via query params:
    ?origin=prose                 → filter by origin
    ?type=concept                 → filter by type
    ?full=1                       → show all (no pagination)
    ?page=2                       → page number for paginated commands

All responses are text/plain — same format agents see on the CLI.
"""

import subprocess
import sys
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

SCRIPT = Path(__file__).parent / "lumens-mirror-explore.py"

app = FastAPI(
    title="Lumen's Mirror API",
    description="HTTP wrapper for the graph explorer CLI. Returns identical output.",
    version="0.1.0",
)


def run_explore(*args):
    """Run the explore script with given arguments, return stdout."""
    cmd = [sys.executable, str(SCRIPT)] + list(args)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=10,
        cwd=str(SCRIPT.parent),
    )
    return result.stdout + result.stderr


@app.get("/", response_class=PlainTextResponse)
def home(origin: str = None, type: str = None, full: str = None):
    """Home page — equivalent to running the script with no args."""
    args = ["explore"]
    if origin:
        args += ["--origin", origin]
    if type:
        args += ["--type", type]
    if full:
        args.append("--full")
    return run_explore(*args)


@app.get("/cmd/{path:path}", response_class=PlainTextResponse)
def command(path: str, origin: str = None, type: str = None, full: str = None, page: str = None):
    """
    Run any explorer command. Path segments become CLI arguments.
    Use -- as path separator for commands that need it (e.g., path/X/--/Y).
    """
    parts = path.split("/")
    args = []
    for part in parts:
        if part:
            args.append(part)

    if origin:
        args += ["--origin", origin]
    if type:
        args += ["--type", type]
    if full:
        args.append("--full")
    if page:
        args.append(page)

    return run_explore(*args)


@app.get("/help", response_class=PlainTextResponse)
def help_text():
    """Show CLI help/usage."""
    return run_explore("help")
