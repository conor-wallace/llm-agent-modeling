from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from collections import Counter

# ---------- Channel map per your spec ----------
CH = {
    "self_pos": 0,
    "other_pos": 1,
    "self_orient": slice(2, 6),
    "other_orient": slice(6, 10),
    "pots": 10,
    "counters": 11,
    "onion_piles": 12,
    "tomato_piles": 13,
    "plate_piles": 14,
    "delivery": 15,
    "onions_in_pot": 16,     # non-binary
    "tomatoes_in_pot": 17,   # non-binary
    "onions_in_soup": 18,    # non-binary
    "tomatoes_in_soup": 19,  # non-binary
    "cook_time": 20,         # 19..1 while cooking
    "soup_done": 21,         # 1 at done pots and at soup items on tiles
    "plates": 22,            # loose items
    "onions": 23,
    "tomatoes": 24,
    "urgency": 25
}

ACTIONS = ["UP","DOWN","LEFT","RIGHT","STAY","INTERACT"]


# ---------- low-level helpers ----------
def coords(layer: np.ndarray, thresh: float = 0.5) -> List[Tuple[int, int]]:
    ys, xs = np.where(layer > thresh)
    return [(int(x), int(y)) for y, x in zip(ys, xs)]

def single_coord(layer: np.ndarray, thresh: float = 0.5) -> Optional[Tuple[int, int]]:
    pts = coords(layer, thresh)
    return pts[0] if pts else None

def val_at(layer: np.ndarray, pos: Tuple[int,int]) -> float:
    x, y = pos
    return float(layer[y, x])

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def near(pos: Tuple[int,int], targets: List[Tuple[int,int]], r: int = 1) -> bool:
    return any(manhattan(pos, t) <= r for t in targets)


# ---------- station extraction from one observation ----------
def extract_static_stations(obs: np.ndarray) -> Dict[str, List[Tuple[int,int]]]:
    pot_tiles   = coords(obs[:,:,CH["pots"]])
    plate_piles_tiles = coords(obs[:,:,CH["plate_piles"]])
    onion_piles_tiles = coords(obs[:,:,CH["onion_piles"]])
    window_tiles= coords(obs[:,:,CH["delivery"]])
    return {
        "pot_tiles": pot_tiles,
        "plate_piles_tiles": plate_piles_tiles,
        "onion_piles_tiles": onion_piles_tiles,
        "window_tiles": window_tiles
    }

def teammate_pos_from_obs(obs: np.ndarray, teammate_is_other: bool = True) -> Optional[Tuple[int,int]]:
    layer = CH["other_pos"] if teammate_is_other else CH["self_pos"]
    return single_coord(obs[:,:,layer])

def ready_pot_tiles(obs: np.ndarray) -> List[Tuple[int,int]]:
    """Tiles that are pots with soup_done=1 (ready/done soups on pot)."""
    pots = set(coords(obs[:,:,CH["pots"]]))
    done = set(coords(obs[:,:,CH["soup_done"]]))
    return sorted(list(pots & done))

def soup_on_counters_tiles(obs: np.ndarray) -> List[Tuple[int,int]]:
    """Done soups on counters (not on pot tiles)."""
    pots = set(coords(obs[:,:,CH["pots"]]))
    done = set(coords(obs[:,:,CH["soup_done"]]))
    return sorted(list(done - pots))


# ---------- per-step feature extraction ----------
def step_context_features(
    obs: np.ndarray,
    teammate_action: str,
    teammate_reward: float,
    teammate_is_other: bool = True,
    near_radius: int = 1
) -> Dict[str, Any]:
    """
    Extract per-step signals used for K-step aggregates.
    """
    assert obs.ndim == 3 and obs.shape[2] == 26, "Expected (H, W, 26) observation"
    stations = extract_static_stations(obs)
    pot_tiles = stations["pot_tiles"]
    plate_piles_tiles = stations["plate_piles_tiles"]
    onion_piles_tiles = stations["onion_piles_tiles"]
    window_tiles = stations["window_tiles"]

    tm_pos = teammate_pos_from_obs(obs, teammate_is_other=teammate_is_other)
    if tm_pos is None:
        # No visible teammate (shouldn't happen often) — return neutral features
        return dict(
            pos=None,
            action=teammate_action,
            reward=teammate_reward,
            near_pot=False, near_plate=False, near_window=False,
            on_ready_pot=False, on_done_soup=False,
            interact_at_pot=False, interact_at_plate=False, interact_at_ready_or_soup=False
        )

    # proximity
    near_pot   = near(tm_pos, pot_tiles, r=near_radius)
    near_plate_pile = near(tm_pos, plate_piles_tiles, r=near_radius)
    near_onion_pile = near(tm_pos, onion_piles_tiles, r=near_radius)
    near_win   = near(tm_pos, window_tiles, r=near_radius)

    # readiness contexts
    ready_pots   = ready_pot_tiles(obs)
    done_soups   = soup_on_counters_tiles(obs)
    on_ready_pot = tm_pos in ready_pots
    on_done_soup = tm_pos in done_soups  # soup dish on a counter

    # interaction classification by tile
    interact = (teammate_action == "INTERACT")
    interact_at_pot   = interact and (tm_pos in pot_tiles)
    interact_at_plate = interact and (tm_pos in plate_piles_tiles)
    interact_at_onion = interact and (tm_pos in onion_piles_tiles)
    interact_at_ready_or_soup = interact and (tm_pos in set(ready_pots) | set(done_soups))

    return dict(
        pos=tm_pos,
        action=teammate_action,
        reward=teammate_reward,
        near_pot=near_pot,
        near_plate=near_plate_pile,
        near_onion=near_onion_pile,
        near_window=near_win,
        on_ready_pot=on_ready_pot,
        on_done_soup=on_done_soup,
        interact_at_pot=interact_at_pot,
        interact_at_plate=interact_at_plate,
        interact_at_onion=interact_at_onion,
        interact_at_ready_or_soup=interact_at_ready_or_soup
    )


# ---------- rolling-window (K-step) fingerprint extraction ----------
def dwell_streak(flags: List[bool]) -> int:
    """Longest consecutive True streak."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best

def time_to_first(flag_events: List[bool]) -> int:
    """Index of first True, else a large sentinel."""
    for i, v in enumerate(flag_events):
        if v: return i
    return -1  # ∞ sentinel

def action_ngrams(actions: List[str], n: int = 2, top_k: int = 3) -> List[str]:
    seq = [a for a in actions if a in ACTIONS]
    grams = ["→".join(seq[i:i+n]) for i in range(len(seq)-n+1)]
    c = Counter(grams)
    return [g for g,_ in c.most_common(top_k)]

def infer_first_destination(near_pot_seq: List[bool], near_plate_seq: List[bool], near_onion_seq: List[bool], near_window_seq: List[bool]) -> str:
    for i in range(len(near_pot_seq)):
        if near_pot_seq[i]:   return "pot"
        if near_plate_seq[i]: return "plate_pile"
        if near_onion_seq[i]: return "onion_pile"
        if near_window_seq[i]:return "window"
    return "unknown"

def held_item_proxy(history: List[Dict[str,Any]]) -> Dict[str, Any]:
    """
    Very rough heuristic:
    - If we see INTERACT at plate_pile and subsequently the agent leaves that tile,
      guess 'plate' for a few steps.
    - Mark as guess=True; do not over-rely on this.
    """
    holding = None
    guess = False
    # Look back for a recent plate interaction and departure
    for i in range(len(history)-1, -1, -1):
        h = history[i]
        if h["interact_at_plate"]:
            # If it moved away after this point, we assume picked up a plate
            if any(not history[j]["pos"] == history[i]["pos"] for j in range(i+1, len(history))):
                holding = "plate"; guess = True
                break
    return {"value": holding, "guess": guess}

def blocked_event_proxy(obs: np.ndarray, h: Dict[str,Any]) -> int:
    """
    Best-effort 'blocked' detection:
    - INTERACT at pot when it's not ready and onions_in_pot layer at that tile is 0
      could indicate a failed scoop (or incorrect timing). This is a weak heuristic.
    """
    if not h["interact_at_pot"]:
        return 0
    pos = h["pos"]
    if pos is None: return 0
    onions_in_pot = obs[:,:,CH["onions_in_pot"]]
    cook_time     = obs[:,:,CH["cook_time"]]
    soup_done     = obs[:,:,CH["soup_done"]]

    # Blocked if trying to interact when pot is neither cooking nor done and empty
    empty = (val_at(onions_in_pot, pos) <= 0.5) and (val_at(cook_time, pos) <= 0.5) and (val_at(soup_done, pos) <= 0.5)
    return int(empty)

def handoff_event_proxy(window: List[Dict[str,Any]]) -> int:
    """
    Heuristic 'handoff' event:
    - After plate-pile interaction, the agent waits near pot/window >=2 steps (likely holding plate).
    """
    # Look for a pattern: plate interact -> near pot or window for 2+ consecutive steps
    consecutive = 0
    after_plate = False
    count = 0
    for h in window:
        if h["interact_at_plate"]:
            after_plate = True
            consecutive = 0
        if after_plate and (h["near_pot"] or h["near_window"]):
            consecutive += 1
            if consecutive >= 2:
                count += 1
                after_plate = False
                consecutive = 0
        elif after_plate:
            consecutive = 0
    return count


def summarize_results(results: np.ndarray, labels: np.ndarray, columns: list[str]):
    """
    Compute mean and std across multiple runs for each class.

    Parameters
    ----------
    results : np.ndarray
        Shape (n_runs, n_classes). Each row = per-class accuracy (or any metric) for one run.
    labels : list of str
        Names of the classes, length = n_classes.

    Returns
    -------
    summary : pd.DataFrame
        DataFrame with mean, std, and mean±std for each class and overall.
    """

    # results is shape (N, M) and labels is shape (N,)
    x = np.concatenate([results, labels[:, np.newaxis]], axis=1)

    df = pd.DataFrame(x, columns=columns)

    teammate_types = df["teammate_type"].unique()

    teammate_summaries = {}
    for teammate_type in teammate_types:
        subset = df[df["teammate_type"] == teammate_type]
        # Get the mean and std for each feature except for 'teammate_type'
        means = subset.mean(axis=0)
        stds = subset.std(axis=0)

        summary = {}

        for f, m, s in zip(subset.columns, means, stds):
            summary.update({
                f + '_mean': m,
                f + '_std': s,
                f + '_mean±std': f"{m:.2f} ± {s:.2f}"
            })

        teammate_summaries[teammate_type] = summary

    return teammate_summaries


def fingerprint_from_window(
    obs_window: List[np.ndarray],
    teammate_actions: List[str],
    teammate_rewards: List[float],
    teammate_is_other: bool = True,
    near_radius: int = 1
) -> Dict[str, Any]:
    """
    Compute the K-step fingerprint features from a list of observations and teammate actions of equal length.
    """
    assert len(obs_window) == len(teammate_actions) and len(obs_window) > 0
    K = len(obs_window)

    # per-step contexts
    step_feats: List[Dict[str,Any]] = [
        step_context_features(o, a, r, teammate_is_other=teammate_is_other, near_radius=near_radius)
        for o, a, r in zip(obs_window, teammate_actions, teammate_rewards)
    ]

    near_pot_seq    = [h["near_pot"] for h in step_feats]
    near_plate_seq  = [h["near_plate"] for h in step_feats]
    near_onion_seq  = [h["near_onion"] for h in step_feats]
    near_window_seq = [h["near_window"] for h in step_feats]
    act_seq         = [h["action"] for h in step_feats]
    reward_seq      = [h["reward"] for h in step_feats]

    # Counts
    near_pot_steps    = int(sum(near_pot_seq))
    near_plate_steps  = int(sum(near_plate_seq))
    near_onion_steps  = int(sum(near_onion_seq))
    near_window_steps = int(sum(near_window_seq))

    interacts_with_pot   = int(sum(h["interact_at_pot"] for h in step_feats))
    interacts_with_plate = int(sum(h["interact_at_plate"] for h in step_feats))
    interacts_with_onion = int(sum(h["interact_at_onion"] for h in step_feats))
    interacts_at_ready_or_soup = int(sum(h["interact_at_ready_or_soup"] for h in step_feats))

    unique_tasks_attempted = int(interacts_with_pot > 0) + int(interacts_with_plate > 0) + int(interacts_with_onion > 0) + int(interacts_at_ready_or_soup > 0)

    cumulative_reward = sum(reward_seq)

    # Timing
    t_first_pot   = time_to_first([h["interact_at_pot"] for h in step_feats])
    t_first_plate = time_to_first([h["interact_at_plate"] for h in step_feats])
    t_first_onion = time_to_first([h["interact_at_onion"] for h in step_feats])
    t_first_serve = time_to_first([h["interact_at_ready_or_soup"] for h in step_feats])

    # Dwell (longest streak)
    dwell_pot   = dwell_streak(near_pot_seq)
    dwell_plate = dwell_streak(near_plate_seq)
    dwell_onion = dwell_streak(near_onion_seq)
    dwell_window= dwell_streak(near_window_seq)

    # First destination
    first_destination = infer_first_destination(near_pot_seq, near_plate_seq, near_onion_seq, near_window_seq)
    first_destination_idx = {"pot": 0, "plate": 1, "onion": 2, "window": 3}.get(first_destination, -1)

    # N-grams
    ngrams2 = action_ngrams(act_seq, n=2, top_k=3)

    # Heuristic extras
    held_guess = held_item_proxy(step_feats)
    # blocked / handoff: need observation for each step to judge 'blocked' (use most recent obs here for a rough count)
    blocked_events = sum(blocked_event_proxy(obs_window[i], step_feats[i]) for i in range(K))
    handoff_events = handoff_event_proxy(step_feats)

    # Phase (optional): simple cut by index — you can pass absolute t to compute a better phase
    phase = "opening"  # adjust as needed outside this function

    # Assemble
    return {
        "cumulative_reward": cumulative_reward,
        "first_destination": first_destination_idx,
        "near_pot_steps": near_pot_steps,
        "near_plate_pile_steps": near_plate_steps,
        "near_onion_pile_steps": near_onion_steps,
        "near_window_steps": near_window_steps,
        "interacts_with_pot": interacts_with_pot,
        "interacts_with_plate_pile": interacts_with_plate,
        "interacts_with_onion_pile": interacts_with_onion,
        "interacts_at_ready_pot_or_soup": interacts_at_ready_or_soup,
        "unique_tasks_attempted_in_first_K": unique_tasks_attempted,
        "time_to_first_interact_pot": t_first_pot,
        "time_to_first_interact_plate": t_first_plate,
        "time_to_first_interact_onion": t_first_onion,
        "time_to_first_interact_serve": t_first_serve,
        "dwell_pot": dwell_pot,
        "dwell_plate": dwell_plate,
        "dwell_onion": dwell_onion,
        "dwell_window": dwell_window,
        "blocked_events": blocked_events,     # heuristic
        "handoff_events": handoff_events      # heuristic
    }


class FingerprintVectorizer:
    """
    Fit/transform fingerprints (dicts) from `fingerprint_from_window` into
    fixed-length numeric vectors for MI or linear probes.

    Usage:
      vec = FingerprintVectorizer(use_ngram_vocab=True, normalize_by_K=True)
      vec.fit(list_of_fingerprints)              # builds one-hot/NG vocab
      X = vec.transform(list_of_fingerprints)    # -> (N, D) float32
      names = vec.feature_names_                 # list of D feature names
    """

    # Base numeric features pulled directly from the fingerprint dict
    _NUMERIC_KEYS = [
        # counts over last K steps
        "near_pot_steps",
        "near_plate_pile_steps",
        "near_window_steps",
        "interacts_with_pot",
        "interacts_with_plate_pile",
        "interacts_at_ready_pot_or_soup",
        "unique_tasks_attempted_in_first_K",
        # timings (use large sentinel handling)
        "time_to_first_interact_pot",
        "time_to_first_interact_plate",
        "time_to_first_interact_serve",
        # dwell streaks
        "dwell_pot",
        "dwell_plate",
        "dwell_window",
        # extras
        "blocked_events",
        "handoff_events",
    ]

    # Categorical one-hot for first destination
    _DEST_CATS = ["pot", "plate_pile", "window", "other", "unknown"]

    def __init__(self,
                 use_ngram_vocab: bool = False,
                 ngram_vocab_size: int = 20,
                 normalize_by_K: bool = True):
        self.use_ngram_vocab = use_ngram_vocab
        self.ngram_vocab_size = ngram_vocab_size
        self.normalize_by_K = normalize_by_K

        self.dest_index_: Dict[str, int] = {c: i for i, c in enumerate(self._DEST_CATS)}
        self.ngram_vocab_: Dict[str, int] = {}  # filled in fit if use_ngram_vocab
        self.feature_names_: List[str] = []

    @staticmethod
    def _safe_get(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
        v = d.get(key, default)
        try:
            return float(v)
        except Exception:
            return default

    def _build_feature_names(self):
        names = []
        # numeric (optionally normalized)
        for k in self._NUMERIC_KEYS:
            names.append(k + ("/K" if self.normalize_by_K and "time_to_first" not in k else ""))

        # one-hot: first_destination
        for c in self._DEST_CATS:
            names.append(f"first_destination=={c}")

        # held item (proxy)
        names.append("held_item_is_plate_proxy")
        names.append("held_item_proxy_flag")  # whether it was a guess

        # optional ngram vocab
        if self.use_ngram_vocab and self.ngram_vocab_:
            for g in sorted(self.ngram_vocab_, key=self.ngram_vocab_.get):
                names.append(f"ngram::{g}")

        self.feature_names_ = names

    def fit(self, fingerprints: List[Dict[str, Any]]):
        """
        Learn the n-gram vocabulary (if enabled). One-hot categories are fixed.
        """
        if self.use_ngram_vocab:
            counts: Dict[str, int] = {}
            for fp in fingerprints:
                grams = fp.get("action_ngrams_top3", []) or []
                for g in grams:
                    counts[g] = counts.get(g, 0) + 1
            # top n most frequent
            top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[: self.ngram_vocab_size]
            self.ngram_vocab_ = {g: i for i, (g, _) in enumerate(top)}
        self._build_feature_names()
        return self

    def _transform_one(self, fp: Dict[str, Any]) -> np.ndarray:
        K = int(fp.get("K", 10)) or 10
        invK = 1.0 / max(1, K)

        # 1) numeric features (some optionally normalized)
        vec: List[float] = []
        for k in self._NUMERIC_KEYS:
            val = self._safe_get(fp, k, 0.0)
            if self.normalize_by_K and ("time_to_first" not in k):
                val = val * invK
            # clamp the "infinite" sentinel if present
            if "time_to_first" in k and val >= 1_000_000:
                # represent "no event" with K+1 (or 1.5*K after normalization)
                val = (K + 1) if not self.normalize_by_K else (K + 1) * invK
            vec.append(val)

        # 2) one-hot for first_destination
        dest = fp.get("first_destination", "unknown")
        dest = dest if dest in self._DEST_CATS else "other"
        onehot = [0.0] * len(self._DEST_CATS)
        onehot[self.dest_index_[dest]] = 1.0
        vec.extend(onehot)

        # 3) held item proxy
        held = fp.get("held_item", {}) or {}
        held_val = 1.0 if held.get("value") == "plate" else 0.0
        held_guess = 1.0 if bool(held.get("guess")) else 0.0
        vec.append(held_val)
        vec.append(held_guess)

        # 4) optional n-gram counts (binary presence is fine)
        if self.use_ngram_vocab and self.ngram_vocab_:
            grams = set(fp.get("action_ngrams_top3", []) or [])
            for g in sorted(self.ngram_vocab_, key=self.ngram_vocab_.get):
                vec.append(1.0 if g in grams else 0.0)

        return np.asarray(vec, dtype=np.float32)

    def transform(self, fingerprints: List[Dict[str, Any]]) -> np.ndarray:
        if not self.feature_names_:
            self._build_feature_names()
        X = np.vstack([self._transform_one(fp) for fp in fingerprints])
        return X

    # convenience for MI experiments
    def fit_transform(self, fingerprints: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        self.fit(fingerprints)
        X = self.transform(fingerprints)
        return X, self.feature_names_