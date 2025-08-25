import numpy as np
from typing import Any, List, Tuple, Optional, Dict
from collections import Counter

from jaxmarl.environments.overcooked.common import (
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
    OBJECT_INDEX_TO_VEC,
    DIR_TO_VEC
)

# --- channels (same as before) ---
CH = {
    "self_pos": 0, "other_pos": 1,
    "self_orient": slice(2, 6), "other_orient": slice(6, 10),
    "pots": 10, "counters": 11, "onion_piles": 12, "tomato_piles": 13,
    "plate_piles": 14, "delivery": 15, "onions_in_pot": 16, "tomatoes_in_pot": 17,
    "onions_in_soup": 18, "tomatoes_in_soup": 19, "cook_time": 20, "soup_done": 21,
    "plates": 22, "onions": 23, "tomatoes": 24, "urgency": 25
}

ACTIONS = ["UP","DOWN","LEFT","RIGHT","STAY","INTERACT"]

def _coords(layer: np.ndarray, thresh: float = 0.5) -> List[Tuple[int,int]]:
    ys, xs = np.where(layer > thresh)
    return [(int(x), int(y)) for y, x in zip(ys, xs)]

def _single(layer: np.ndarray, thresh: float = 0.5) -> Optional[Tuple[int,int]]:
    pts = _coords(layer, thresh); return pts[0] if pts else None

def _manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def _near(pos: Tuple[int,int], targets: List[Tuple[int,int]], r: int = 1) -> bool:
    return any(_manhattan(pos, t) <= r for t in targets)

def _stations(obs: np.ndarray) -> Dict[str, List[Tuple[int,int]]]:
    return {
        "pots": _coords(obs[:,:,CH["pots"]]),
        "plates": _coords(obs[:,:,CH["plate_piles"]]),
        "window": _coords(obs[:,:,CH["delivery"]]),
        "onion": _coords(obs[:,:,CH["onion_piles"]]),
    }

def _teammate_pos(obs: np.ndarray, teammate_is_other: bool=True) -> Optional[Tuple[int,int]]:
    layer = CH["other_pos"] if teammate_is_other else CH["self_pos"]
    return _single(obs[:,:,layer])

def _first_destination_from_dwell(dwell: Dict[str,int]) -> str:
    # first station with dwell >= 2 is considered a destination
    order = ["plates","pots","window","onion"]
    for name in order:
        if dwell.get(name, 0) >= 2: return {"plates":"plate_pile"}.get(name, name)
    return "unknown"


def _behavioral_stats_from_history(
    obs_window: List[np.ndarray],
    teammate_actions: List[str],
    teammate_is_other: bool=True,
    near_radius: int=1
) -> Dict[str, Any]:
    assert len(obs_window) == len(teammate_actions) and len(obs_window) > 0
    K = len(obs_window)
    # accumulators
    near_counts = Counter({"plates":0,"pots":0,"window":0,"onion":0})
    dwell_streaks = {"plates":0,"pots":0,"window":0,"onion":0}
    cur_streak = {"plates":0,"pots":0,"window":0,"onion":0}
    near_seq = []

    for obs, act in zip(obs_window, teammate_actions):
        S = _stations(obs)
        pos = _teammate_pos(obs, teammate_is_other)
        flags = {"plates": False, "pots": False, "window": False, "onion": False}
        if pos is not None:
            flags["plates"] = _near(pos, S["plates"], near_radius)
            flags["pots"]   = _near(pos, S["pots"], near_radius)
            flags["window"] = _near(pos, S["window"], near_radius)
            flags["onion"]  = _near(pos, S["onion"], near_radius)
        # count and streaks
        for k,v in flags.items():
            if v:
                near_counts[k] += 1
                cur_streak[k] += 1
                dwell_streaks[k] = max(dwell_streaks[k], cur_streak[k])
            else:
                cur_streak[k] = 0
        # compact per-step tuple for the prompt
        near_label = next((n for n in ["plates","pots","window","onion"] if flags[n]), "none")
        near_seq.append((act if act in ACTIONS else "STAY", near_label))

    # normalized ratios
    norm = max(1, K)
    ratios = {k: near_counts[k]/norm for k in near_counts}
    first_dest = _first_destination_from_dwell(dwell_streaks)

    # Optional: ready-pot serve interactions (low-MI early but harmless)
    soup_done = _coords(obs_window[-1][:,:,CH["soup_done"]])
    tmpos_last = _teammate_pos(obs_window[-1], teammate_is_other)
    interact_last = 1 if (teammate_actions[-1] == "INTERACT" and tmpos_last in soup_done) else 0

    return {
        "K": K,
        "near_plate_pile_steps": int(near_counts["plates"]),
        "near_pot_steps": int(near_counts["pots"]),
        "near_window_steps": int(near_counts["window"]),
        "near_onion_pile_steps": int(near_counts["onion"]),
        "dwell_plate": int(dwell_streaks["plates"]),
        "dwell_pot": int(dwell_streaks["pots"]),
        "dwell_window": int(dwell_streaks["window"]),
        "dwell_onion": int(dwell_streaks["onion"]),
        "station_time_ratio": [ratios["plates"], ratios["pots"], ratios["window"], ratios["onion"]],
        "first_destination": first_dest,
        "interacts_at_ready_pot_or_soup": int(interact_last),
        "behavior_window": near_seq  # list of (action, near_label) for last K steps
    }


def get_other_pos(obs: np.ndarray) -> tuple[int, int]:
    """Get the position of the other agent from the observation."""
    other_pos = _single_coord(obs[:, :, CH["other_pos"]])
    return other_pos if other_pos is not None else (0, 0)


def get_agent_holding(agent_inv_idx):
    for name, val in OBJECT_TO_INDEX.items():
        if val == agent_inv_idx:
            return name
    return "unknown"


def describe_overcooked_obs(
    obs: np.ndarray,
    layout_id: Optional[str] = None,
    time_left: Optional[int] = None,
    onions_needed: int = 3
) -> str:
    """
    Convert an Overcooked (H, W, 26) tensor into a concise natural-language description
    suitable for prompting an LLM teammate-model. Does not guess fields not present
    in the tensor (e.g., held items).
    """
    assert obs.ndim == 3 and obs.shape[2] == 26, "Expected (H, W, 26) observation"

    H, W, _ = obs.shape
    # Agent positions
    self_pos = _single_coord(obs[:, :, CH["self_pos"]])
    other_pos = _single_coord(obs[:, :, CH["other_pos"]])

    # Orientations
    self_dir_idx = _one_hot_dir(obs[:, :, CH["self_orient"]])
    other_dir_idx = _one_hot_dir(obs[:, :, CH["other_orient"]])

    # Static objects
    pot_tiles = _coords(obs[:, :, CH["pots"]])
    counter_tiles = _coords(obs[:, :, CH["counters"]])
    onion_piles = _coords(obs[:, :, CH["onion_piles"]])
    plate_piles = _coords(obs[:, :, CH["plate_piles"]])
    delivery_tiles = _coords(obs[:, :, CH["delivery"]])

    # Variable items on tiles
    loose_plates = _coords(obs[:, :, CH["plates"]])
    loose_onions = _coords(obs[:, :, CH["onions"]])
    loose_tomatoes = _coords(obs[:, :, CH["tomatoes"]])  # likely empty in this env

    # Pot/soup numeric layers
    onions_in_pot = obs[:, :, CH["onions_in_pot"]]
    tomatoes_in_pot = obs[:, :, CH["tomatoes_in_pot"]]
    onions_in_soup = obs[:, :, CH["onions_in_soup"]]
    tomatoes_in_soup = obs[:, :, CH["tomatoes_in_soup"]]
    cook_time = obs[:, :, CH["cook_time"]]
    soup_done = obs[:, :, CH["soup_done"]]  # 1 at done pots and soups on tiles

    # Identify soups on counters (done soups not on pot tiles)
    soup_done_coords = set(_coords(soup_done))
    pot_tile_set = set(pot_tiles)
    soup_on_counters = sorted(list(soup_done_coords - pot_tile_set))

    # Pot status objects
    pots_status: List[Dict] = []
    for p in pot_tiles:
        pot_info = {
            "pos": p,
            "onions_pre_cook": int(round(_value_at(onions_in_pot, p))),
            "tomatoes_pre_cook": int(round(_value_at(tomatoes_in_pot, p))),
            "onions_in_soup": int(round(_value_at(onions_in_soup, p))),
            "tomatoes_in_soup": int(round(_value_at(tomatoes_in_soup, p))),
            "cook_time": int(round(_value_at(cook_time, p))),
            "done": _value_at(soup_done, p) > 0.5
        }
        pots_status.append(pot_info)

    # Urgency
    urgent = obs[:, :, CH["urgency"]].mean() > 0.5  # entire layer is 1 when <=40 steps remain

    # ---- Build the description ----
    lines = []
    if layout_id is not None:
        lines.append(f"Layout: {layout_id}.")
    if time_left is not None:
        lines.append(f"Time left: {int(time_left)} steps.")

    # 1) Agents
    lines.append("Agents:")
    if self_pos is not None:
        self_dir = ORIENT_LABELS.get(self_dir_idx, "UNKNOWN")
        lines.append(f"- Agent_1 at {self_pos}, facing {self_dir}.")
    else:
        lines.append("- Agent_1 position: UNKNOWN.")
    if other_pos is not None:
        other_dir = ORIENT_LABELS.get(other_dir_idx, "UNKNOWN")
        lines.append(f"- Agent_2 at {other_pos}, facing {other_dir}.")
    else:
        lines.append("- Agent_2 position: UNKNOWN.")

    # 2) Static objects
    def _fmt_list(coords_list):
        return "none" if not coords_list else ", ".join(map(str, coords_list))

    lines.append("Static objects:")
    lines.append(f"- Pots at: {_fmt_list(pot_tiles)}.")
    lines.append(f"- Counters at: {_fmt_list(counter_tiles)}.")
    lines.append(f"- Onion pile(s) at: {_fmt_list(onion_piles)}.")
    lines.append(f"- Plate pile(s) at: {_fmt_list(plate_piles)}.")
    lines.append(f"- Delivery window(s) at: {_fmt_list(delivery_tiles)}.")

    # 3) Pot & soup status
    if pots_status:
        lines.append("Pot status:")
        for info in pots_status:
            p = info["pos"]
            pre_on = info["onions_pre_cook"]
            pre_to = info["tomatoes_pre_cook"]
            in_soup_on = info["onions_in_soup"]
            in_soup_to = info["tomatoes_in_soup"]
            ct = info["cook_time"]
            done = info["done"]
            # human-friendly summary
            detail_parts = []
            if pre_on > 0 or pre_to > 0:
                detail_parts.append(f"pre-cook contents: onions={pre_on}, tomatoes={pre_to}")
            if in_soup_on > 0 or in_soup_to > 0:
                detail_parts.append(f"in-soup: onions={in_soup_on}, tomatoes={in_soup_to}")
            if ct > 0:
                detail_parts.append(f"cooking, time remaining={ct}")
            if done:
                detail_parts.append("DONE")
            if not detail_parts:
                detail_parts.append("empty/idle")
            lines.append(f"- Pot at {p}: " + "; ".join(detail_parts) + ".")
    else:
        lines.append("Pot status: none.")

    # 4) Soups on counters (ready to serve)
    lines.append(f"Soups on counters (done, not in pot): {_fmt_list(soup_on_counters)}.")

    # 5) Loose items
    lines.append("Loose items on tiles:")
    lines.append(f"- Plates at: {_fmt_list(loose_plates)}.")
    lines.append(f"- Onions at: {_fmt_list(loose_onions)}.")
    if loose_tomatoes:
        lines.append(f"- Tomatoes at: {_fmt_list(loose_tomatoes)}.")  # likely empty in onion-only env

    # 6) Urgency
    if urgent:
        lines.append("Urgency: 40 or fewer timesteps remain.")

    # 7) Notes / limitations
    lines.append(
        "Note: Held items are not encoded in these observation layers; this description does not infer what an agent is carrying."
    )

    # Optional: add recipe info if needed
    lines.append(f"Recipe requirement: {onions_needed} onions per soup.")

    return "\n".join(lines)


def describe_overcooked_obs_enhanced(
    obs_now: np.ndarray,
    layout_id: Optional[str],
    time_left: Optional[int],
    obs_history: List[np.ndarray],
    teammate_actions_history: List[str],
    K: int = 10,
    teammate_is_other: bool = True,
    phase: str = "probe",
    probe_policy_note: Optional[str] = "First K steps use a fixed probe policy; classify the teammate’s response.",
    onions_needed: int = 3
) -> str:
    """
    Returns a scene string optimized for CoLLAB/ReCoLLAB:
    - Phase header + probe note
    - BehavioralStats (last K steps): high-MI proximity/dwell features
    - Teammate_behavior_window (K tuples)
    - Static objects (pots/plates/window/onion)
    """
    assert obs_now.ndim == 3 and obs_now.shape[2] == 26
    # pick the last K
    obs_window = obs_history[-K:] if len(obs_history) >= K else obs_history
    act_window = teammate_actions_history[-K:] if len(teammate_actions_history) >= K else teammate_actions_history
    stats = _behavioral_stats_from_history(obs_window, act_window, teammate_is_other=teammate_is_other)

    # current static layout
    S = _stations(obs_now)
    tm_pos = _teammate_pos(obs_now, teammate_is_other)

    # --- build text ---
    lines = []
    if layout_id is not None: lines.append(f"Layout: {layout_id}.")
    if time_left is not None: lines.append(f"Time left: {int(time_left)} steps.")
    lines.append(f"Phase: {phase}.")
    if probe_policy_note: lines.append(f"Probe: {probe_policy_note}")

    lines.append("\nBehavioralStats (last K steps)")
    lines.append(f"K={stats['K']}")
    # High-MI features first
    lines.append(f"near_plate_pile_steps={stats['near_plate_pile_steps']}, near_pot_steps={stats['near_pot_steps']}, near_window_steps={stats['near_window_steps']}, near_onion_pile_steps={stats['near_onion_pile_steps']}")
    lines.append(f"dwell_plate={stats['dwell_plate']}, dwell_pot={stats['dwell_pot']}, dwell_window={stats['dwell_window']}, dwell_onion={stats['dwell_onion']}")
    r = stats["station_time_ratio"]
    lines.append(f"station_time_ratio=[plate={r[0]:.2f}, pot={r[1]:.2f}, window={r[2]:.2f}, onion={r[3]:.2f}]")
    lines.append(f"first_destination={stats['first_destination']}")
    lines.append(f"interacts_at_ready_pot_or_soup={stats['interacts_at_ready_pot_or_soup']}")

    lines.append("\nTeammate_behavior_window (most recent last)")
    # print short tuples only
    for i, (a, near_lbl) in enumerate(stats["behavior_window"]):
        t = i - stats["K"]  # negative index style (e.g., t=-10..-1)
        lines.append(f"- t={t}: (action={a}, near={near_lbl})")

    # concise static info (after behavioral)
    lines.append("\nStatic objects")
    lines.append(f"- Pots: {S['pots'] or 'none'}; Plate piles: {S['plates'] or 'none'}; Window: {S['window'] or 'none'}; Onion piles: {S['onion'] or 'none'}.")
    lines.append(f"- Teammate current pos: {tm_pos or 'unknown'}")
    lines.append(f"Recipe requirement: {onions_needed} onions per soup.")

    return "\n".join(lines)