#!/usr/bin/env python3
"""
Build semantic search index for CVPR Workshop Radar.

Embeds each workshop/tutorial using Ollama's nomic-embed-text model and
writes search_index.json alongside the main data file.

Usage:
    python3 build_search_index.py
    python3 build_search_index.py --model mxbai-embed-large
    python3 build_search_index.py --model nomic-embed-text

Requires Ollama running: `ollama serve`
Then pull the model if needed: `ollama pull nomic-embed-text`
"""

import argparse
import json
import sys
from pathlib import Path

import requests

JSON_PATH  = Path(__file__).parent / "cvpr2026_workshops_tutorials.json"
INDEX_PATH = Path(__file__).parent / "search_index.json"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
DEFAULT_MODEL = "nomic-embed-text"


def check_ollama(model: str) -> None:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        sys.exit(f"Cannot reach Ollama at localhost:11434 — {e}\nRun: ollama serve")
    # Accept both exact name and name:latest
    if model not in names and f"{model}:latest" not in names:
        sys.exit(
            f"Model '{model}' not found in Ollama.\n"
            f"Available: {', '.join(names)}\n"
            f"Pull it with: ollama pull {model}"
        )


def embed(text: str, model: str) -> list[float]:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def event_text(e: dict) -> str:
    parts = [e.get("title") or ""]
    if e.get("organizers"):
        parts.append(e["organizers"])
    if e.get("track"):
        parts.append(e["track"])
    if e.get("summary"):
        parts.append(e["summary"])
    # Include speaker/talk names from program but keep it short
    if e.get("program_text"):
        parts.append(e["program_text"][:800])
    return ". ".join(p.strip() for p in parts if p.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama embedding model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    check_ollama(args.model)

    data = json.loads(JSON_PATH.read_text())
    workshops = data.get("workshops", [])
    tutorials = data.get("tutorials", [])

    total = len(workshops) + len(tutorials)
    print(f"Model : {args.model}")
    print(f"Events: {len(workshops)} workshops + {len(tutorials)} tutorials = {total} total\n")

    index = []
    errors = 0

    for prefix, events in [("w", workshops), ("t", tutorials)]:
        for i, e in enumerate(events):
            event_id = f"{prefix}{i}"
            title = (e.get("title") or "?")[:60]
            print(f"  [{len(index)+1:3d}/{total}] {event_id}  {title}")
            text = event_text(e)
            try:
                embedding = embed(text, args.model)
                index.append({"id": event_id, "embedding": embedding})
            except Exception as exc:
                print(f"           ERROR: {exc}")
                errors += 1

    INDEX_PATH.write_text(json.dumps(index, separators=(",", ":")))
    dim = len(index[0]["embedding"]) if index else 0
    size_kb = INDEX_PATH.stat().st_size // 1024

    print(f"\nSaved → {INDEX_PATH}")
    print(f"  Entries   : {len(index)}/{total}  (errors: {errors})")
    print(f"  Dimensions: {dim}")
    print(f"  File size : {size_kb} KB")


if __name__ == "__main__":
    main()
