from dataclasses import dataclass, field
from typing import Optional, Dict, List

@dataclass
class Signal:
    mode: str           # "PRNG" | "MANIPULATION" | "UNKNOWN" | "ENSEMBLE"
    suggestion: str     # "BET_RED" | "BET_GREEN" | "BET_VIOLET" | "BET_NUMBER_X" | "SKIP"
    confidence: float   # 0.0 - 1.0
    reason: str
    # Optional extras for richer alerts (backward compatible)
    probs: Dict[str, float] = field(default_factory=dict)  # e.g., {"RED":0.64,"GREEN":0.28,"VIOLET":0.08}
    sources: List[str] = field(default_factory=list)       # e.g., ["MarkovColor","Cycle","MarkovNumber"]

def color_for_number(n: int) -> str:
    if n in (0,5): return "VIOLET"
    return "GREEN" if (n % 2 == 0) else "RED"

def suggest_next(pred_next_num: Optional[int], prng_active: bool, imbalance_alert: bool) -> Signal:
    if not prng_active or imbalance_alert:
        return Signal("MANIPULATION", "SKIP", 0.2, "Manipulation indicators active; avoid round.")
    if pred_next_num is None:
        return Signal("UNKNOWN", "SKIP", 0.3, "No reliable cycle found.")
    col = color_for_number(pred_next_num)
    return Signal("PRNG", f"BET_{col}", 0.65, "Cycle stable; using predicted next number → color.")

def suggest_from_ensemble(
    color_probs: Dict[str, float],
    number_vote: Optional[int],
    sources: List[str],
    min_color_prob: float = 0.62,
    min_sources_agree: int = 2,
) -> Signal:
    """Combine color probabilities with optional number vote.

    - color_probs: dict with keys RED/GREEN/VIOLET summing ~1
    - number_vote: optional 0-9 prediction; may be None
    - If top color prob < threshold or not enough sources, return SKIP.
    """
    if not color_probs:
        return Signal("ENSEMBLE", "SKIP", 0.3, "No color probabilities provided.")

    # Determine top color
    top_color, top_p = max(color_probs.items(), key=lambda kv: kv[1])

    # Count agreeing sources (very simple: presence of at least min_sources_agree evidence)
    agree_count = len(set(sources))
    if top_p < min_color_prob or agree_count < min_sources_agree:
        return Signal(
            "ENSEMBLE",
            "SKIP",
            max(0.3, top_p),
            f"Insufficient confidence: top_color={top_color} p={top_p:.2f}, sources={agree_count}/{min_sources_agree}",
            probs=color_probs,
            sources=sources,
        )

    # If number vote exists and maps to the same color, bump confidence slightly
    conf = top_p
    reason_bits = [f"top_color={top_color} p={top_p:.2f}"]
    if number_vote is not None:
        num_color = color_for_number(number_vote)
        reason_bits.append(f"number_vote={number_vote}→{num_color}")
        if num_color == top_color:
            conf = min(0.99, conf + 0.03)

    return Signal(
        "ENSEMBLE",
        f"BET_{top_color}",
        conf,
        ", ".join(reason_bits),
        probs=color_probs,
        sources=sources,
    )
