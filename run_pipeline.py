#!/usr/bin/env python3
"""Fantasy Football Draft Analyzer — main pipeline runner.

Usage:
    python run_pipeline.py [--pick N] [--scoring FILE] [--roster FILE]

Runs all four stages: fetch -> score -> project -> rank.
Stages are independent -- run individually if needed:
    python fetch.py
    python score.py
    python project.py
    python rank.py --pick N
"""

from config import load_config, parse_args
from fetch import fetch_raw_data, fetch_team_defense
from score import run_score_pipeline
from project import run_projection_pipeline
from rank import run_rank_pipeline
from scoring_engine import parse_scoring_rules


def main(argv: list[str] | None = None) -> dict:
    """Run full pipeline. Returns dict of output paths."""
    overrides = parse_args(argv)
    draft_position = overrides.get("pick")
    cfg = load_config(draft_position=draft_position)

    # Override file paths if specified
    if "scoring_file" in overrides:
        cfg["scoring_file"] = overrides["scoring_file"]
    if "roster_file" in overrides:
        cfg["roster_file"] = overrides["roster_file"]

    print("=== Fantasy Football Draft Analyzer ===\n")
    print(f"League: {cfg['num_teams']} teams, {cfg['draft_rounds']} rounds")
    if cfg["draft_position"]:
        print(f"Draft position: #{cfg['draft_position']}")
    print()

    # Stage 1: Fetch
    print("[1/4] Fetching raw data from nflreadpy...")
    fetch_result = fetch_raw_data(cfg)
    for season, count in fetch_result["row_counts"].items():
        print(f"  {season}: {count} rows -> {fetch_result['output_paths'][season]}")

    # Fetch team defense data (separate source: nflreadpy team_stats)
    def_result = fetch_team_defense(cfg)
    for season, count in def_result["row_counts"].items():
        print(f"  {season}: {count} DEF rows -> {def_result['output_paths'][season]}")

    # Stage 2: Score
    print("\n[2/4] Applying scoring rules...")
    rules = parse_scoring_rules(cfg["scoring_file"])
    print(f"  Loaded {len(rules)} scoring rules from {cfg['scoring_file']}")
    score_result = run_score_pipeline(cfg, rules=rules)
    for season, count in score_result["row_counts"].items():
        print(f"  {season}: {count} rows scored")

    # Stage 3: Project
    print("\n[3/4] Running projections...")
    proj_result = run_projection_pipeline(cfg)
    print(f"  {proj_result['player_count']} players projected")
    print(f"  Output: {proj_result['output_path']}")

    # Stage 4: Rank
    print("\n[4/4] Computing rankings...")
    rank_result = run_rank_pipeline(cfg)
    print(f"  Rankings: {rank_result['rankings']}")
    print(f"  Tiers: {rank_result['tiers']}")
    strategy = rank_result.get("strategy")
    if strategy is not None and strategy.exists():
        print(f"  Strategy: {strategy}")
    mock_draft = rank_result.get("mock_draft")
    if mock_draft is not None and mock_draft.exists():
        print(f"  Mock Draft: {mock_draft}")

    print("\n=== Done ===")
    return {
        "fetch": fetch_result,
        "score": score_result,
        "projection": proj_result,
        "rank": rank_result,
    }


if __name__ == "__main__":
    main()
