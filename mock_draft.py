"""Mock draft simulator — full 12-team snake draft with position-aware fills.

Pure functions, no classes. Depends only on polars and roster config.
"""

import polars as pl


def simulate_mock_draft(
    ranked: pl.DataFrame,
    roster: dict,
    num_teams: int = 12,
    rounds: int = 20,
) -> pl.DataFrame:
    """Simulate full snake draft with position-aware fills.

    Each team fills mandatory starters first (by VORP gap), then
    WRT flex (RB/WR/TE), then WRTQ superflex (QB/RB/WR/TE),
    then bench (BPA).

    Args:
        ranked: sorted by overall_rank (VORP descending).
                Columns: player_id, player_name, position, vorp, tier,
                projected_points, overall_rank.
        roster: parsed roster dict from roster.parse_roster().
        num_teams: number of teams in the league.
        rounds: number of draft rounds.

    Returns:
        DataFrame: overall_pick, round, team_slot, player_name,
        player_id, position, vorp, tier, projected_points.
    """
    team_states = {t: _init_team_state(roster) for t in range(1, num_teams + 1)}

    drafted_ids: set[str] = set()
    rows = []
    overall = 0

    for rnd in range(1, rounds + 1):
        order = _get_snake_order(rnd, num_teams)
        for team_slot in order:
            available = ranked.filter(
                ~pl.col("player_id").is_in(list(drafted_ids))
            )

            if len(available) == 0:
                break

            state = team_states[team_slot]
            new_state, _remaining = _pick_for_team(state, available, roster)
            team_states[team_slot] = new_state

            if new_state.get("_skip"):
                continue

            overall += 1

            picked = new_state["drafted"][-1]
            drafted_ids.add(picked["player_id"])

            rows.append({
                "overall_pick": overall,
                "round": rnd,
                "team_slot": team_slot,
                "player_name": picked["player_name"],
                "player_id": picked["player_id"],
                "position": picked["position"],
                "vorp": picked["vorp"],
                "tier": picked["tier"],
                "projected_points": picked["projected_points"],
            })

    return pl.DataFrame(rows)


def _get_snake_order(round_num: int, num_teams: int) -> list[int]:
    """Team order for a round. Odd=forward, Even=reverse."""
    if round_num % 2 == 1:
        return list(range(1, num_teams + 1))
    return list(range(num_teams, 0, -1))


def _init_team_state(roster: dict) -> dict:
    """Initialize empty team state from roster spec.

    Returns dict with:
        needed_starters: list[str]  — flat list of mandatory positions
        needed_flex: list[dict]     — remaining flex slots
        needed_bench: int
        drafted: list[dict]         — players drafted so far
    """
    starters = roster["starters"]
    needed_starters = []
    for pos, count in starters.items():
        needed_starters.extend([pos] * count)

    needed_flex = list(roster["flex"])
    needed_bench = roster["bench"]

    return {
        "needed_starters": needed_starters,
        "needed_flex": needed_flex,
        "needed_bench": needed_bench,
        "drafted": [],
    }


def _needs_starter(team_state: dict) -> list[str]:
    """Return list of unfilled mandatory starter positions."""
    return team_state["needed_starters"]


def _needs_flex(team_state: dict) -> list[dict]:
    """Return list of unfilled flex slots with eligible positions."""
    return team_state["needed_flex"]


def _best_available_at_position(
    available: pl.DataFrame, position: str
) -> float:
    """Return VORP of the best available player at a position.

    Returns -inf if no players available at that position.
    """
    pos_players = available.filter(pl.col("position") == position)
    if len(pos_players) == 0:
        return float("-inf")
    return pos_players["vorp"][0]


def _pick_for_team(
    team_state: dict, available: pl.DataFrame, roster: dict
) -> tuple[dict, pl.DataFrame]:
    """Select best available player for one team's pick.

    Priority cascade:
    1. Mandatory starters (highest VORP gap)
    2. WRT flex (best RB/WR/TE by VORP)
    3. WRTQ superflex (best QB/RB/WR/TE by VORP)
    4. Bench (best overall VORP)

    Returns updated team_state and remaining available DataFrame.
    """
    needed_starters = list(team_state["needed_starters"])
    needed_flex = list(team_state["needed_flex"])
    needed_bench = team_state["needed_bench"]
    new_drafted = list(team_state["drafted"])

    # Priority 1: Fill mandatory starters
    if needed_starters:
        best_pos = None
        best_vorp = float("-inf")
        for pos in set(needed_starters):
            vorp = _best_available_at_position(available, pos)
            if vorp > best_vorp:
                best_vorp = vorp
                best_pos = pos

        if best_pos is not None:
            player = available.filter(
                pl.col("position") == best_pos
            ).head(1)
            if len(player) > 0:
                row = player.row(0, named=True)
                new_drafted.append({
                    "player_name": row["player_name"],
                    "player_id": row["player_id"],
                    "position": row["position"],
                    "vorp": row["vorp"],
                    "tier": row["tier"],
                    "projected_points": row["projected_points"],
                })
                needed_starters.remove(best_pos)

                remaining = available.filter(
                    pl.col("player_id") != row["player_id"]
                )
                return (
                    {
                        "needed_starters": needed_starters,
                        "needed_flex": needed_flex,
                        "needed_bench": needed_bench,
                        "drafted": new_drafted,
                    },
                    remaining,
                )

    # Priority 2: Fill WRT flex (RB/WR/TE)
    if needed_flex:
        for i, slot in enumerate(needed_flex):
            if slot["type"] == "WRT":
                eligible = slot["eligible"]
                flex_players = available.filter(
                    pl.col("position").is_in(eligible)
                )
                if len(flex_players) > 0:
                    row = flex_players.head(1).row(0, named=True)
                    new_drafted.append({
                        "player_name": row["player_name"],
                        "player_id": row["player_id"],
                        "position": row["position"],
                        "vorp": row["vorp"],
                        "tier": row["tier"],
                        "projected_points": row["projected_points"],
                    })
                    del needed_flex[i]

                    remaining = available.filter(
                        pl.col("player_id") != row["player_id"]
                    )
                    return (
                        {
                            "needed_starters": needed_starters,
                            "needed_flex": needed_flex,
                            "needed_bench": needed_bench,
                            "drafted": new_drafted,
                        },
                        remaining,
                    )

    # Priority 3: Fill WRTQ superflex (QB/RB/WR/TE)
    if needed_flex:
        for i, slot in enumerate(needed_flex):
            if slot["type"] == "WRTQ":
                eligible = slot["eligible"]
                flex_players = available.filter(
                    pl.col("position").is_in(eligible)
                )
                if len(flex_players) > 0:
                    row = flex_players.head(1).row(0, named=True)
                    new_drafted.append({
                        "player_name": row["player_name"],
                        "player_id": row["player_id"],
                        "position": row["position"],
                        "vorp": row["vorp"],
                        "tier": row["tier"],
                        "projected_points": row["projected_points"],
                    })
                    del needed_flex[i]

                    remaining = available.filter(
                        pl.col("player_id") != row["player_id"]
                    )
                    return (
                        {
                            "needed_starters": needed_starters,
                            "needed_flex": needed_flex,
                            "needed_bench": needed_bench,
                            "drafted": new_drafted,
                        },
                        remaining,
                    )

    # Priority 4: Fill bench with BPA
    if needed_bench > 0 and len(available) > 0:
        row = available.head(1).row(0, named=True)
        new_drafted.append({
            "player_name": row["player_name"],
            "player_id": row["player_id"],
            "position": row["position"],
            "vorp": row["vorp"],
            "tier": row["tier"],
            "projected_points": row["projected_points"],
        })
        needed_bench -= 1

        remaining = available.filter(
            pl.col("player_id") != row["player_id"]
        )
        return (
            {
                "needed_starters": needed_starters,
                "needed_flex": needed_flex,
                "needed_bench": needed_bench,
                "drafted": new_drafted,
            },
            remaining,
        )

    # Nothing to fill — roster exhausted, pass without picking
    return (
        {
            "needed_starters": needed_starters,
            "needed_flex": needed_flex,
            "needed_bench": needed_bench,
            "drafted": new_drafted,
            "_skip": True,
        },
        available,
    )
