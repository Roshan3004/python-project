from dataclasses import dataclass, field
from typing import Optional, Dict, List

@dataclass
class Signal:
    mode: str           # "MOMENTUM" | "NUMBER_PATTERN" | "TIME_PATTERN" | "ENSEMBLE" | "SKIP"
    suggestion: str     # "BET_RED" | "BET_GREEN" | "BET_VIOLET" | "SKIP"
    confidence: float   # 0.0 - 1.0
    reason: str
    # Optional extras for richer alerts (backward compatible)
    probs: Dict[str, float] = field(default_factory=dict)  # e.g., {"RED":0.64,"GREEN":0.28,"VIOLET":0.08}
    sources: List[str] = field(default_factory=list)       # e.g., ["Momentum","NumberPattern","TimePattern"]

def color_for_number(n: int) -> str:
    if n in (0,5): return "VIOLET"
    return "GREEN" if (n % 2 == 0) else "RED"

def suggest_from_momentum(color_probs: Dict[str, float], min_confidence: float = 0.60) -> Signal:
    """Generate signal based on color momentum analysis"""
    if not color_probs:
        return Signal("SKIP", "SKIP", 0.3, "No color probabilities available")
    
    # Find top color
    top_color = max(color_probs, key=color_probs.get)
    top_prob = color_probs[top_color]
    
    if top_prob >= min_confidence:
        return Signal(
            "MOMENTUM",
            f"BET_{top_color}",
            top_prob,
            f"Strong {top_color} momentum with {top_prob:.2f} probability",
            probs=color_probs,
            sources=["Momentum"]
        )
    else:
        return Signal(
            "SKIP",
            "SKIP",
            top_prob,
            f"Insufficient confidence: top color {top_color} at {top_prob:.2f} < {min_confidence}"
        )

def suggest_from_number_pattern(color_probs: Dict[str, float], min_confidence: float = 0.60) -> Signal:
    """Generate signal based on number pattern correction"""
    if not color_probs:
        return Signal("SKIP", "SKIP", 0.3, "No number pattern data available")
    
    top_color = max(color_probs, key=color_probs.get)
    top_prob = color_probs[top_color]
    
    if top_prob >= min_confidence:
        return Signal(
            "NUMBER_PATTERN",
            f"BET_{top_color}",
            top_prob,
            f"Number pattern correction suggests {top_color} with {top_prob:.2f} probability",
            probs=color_probs,
            sources=["NumberPattern"]
        )
    else:
        return Signal(
            "SKIP",
            "SKIP",
            top_prob,
            f"Weak number pattern: top color {top_color} at {top_prob:.2f} < {min_confidence}"
        )

def suggest_from_time_pattern(color_probs: Dict[str, float], min_confidence: float = 0.60) -> Signal:
    """Generate signal based on time-based patterns"""
    if not color_probs:
        return Signal("SKIP", "SKIP", 0.3, "No time pattern data available")
    
    top_color = max(color_probs, key=color_probs.get)
    top_prob = color_probs[top_color]
    
    if top_prob >= min_confidence:
        return Signal(
            "TIME_PATTERN",
            f"BET_{top_color}",
            top_prob,
            f"Time-based pattern favors {top_color} with {top_prob:.2f} probability",
            probs=color_probs,
            sources=["TimePattern"]
        )
    else:
        return Signal(
            "SKIP",
            "SKIP",
            top_prob,
            f"Weak time pattern: top color {top_color} at {top_prob:.2f} < {min_confidence}"
        )

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

def suggest_next(pred_next_num: Optional[int], prng_active: bool, imbalance_alert: bool) -> Signal:
    """Legacy function for backward compatibility"""
    if not prng_active or imbalance_alert:
        return Signal("MANIPULATION", "SKIP", 0.2, "Manipulation indicators active; avoid round.")
    if pred_next_num is None:
        return Signal("UNKNOWN", "SKIP", 0.3, "No reliable cycle found.")
    col = color_for_number(pred_next_num)
    return Signal("PRNG", f"BET_{col}", 0.65, "Cycle stable; using predicted next number → color.")

def generate_aggressive_signals(color_probs: Dict[str, float], min_confidence: float = 0.55) -> List[Signal]:
    """Generate multiple signals with lower thresholds for more frequent alerts"""
    signals = []
    
    if not color_probs:
        return signals
    
    # Sort colors by probability
    sorted_colors = sorted(color_probs.items(), key=lambda x: x[1], reverse=True)
    
    # Generate signal for top color if above threshold
    top_color, top_prob = sorted_colors[0]
    if top_prob >= min_confidence:
        signals.append(Signal(
            "MOMENTUM",
            f"BET_{top_color}",
            top_prob,
            f"Primary signal: {top_color} momentum at {top_prob:.2f}",
            probs=color_probs,
            sources=["Momentum"]
        ))
    
    # Generate secondary signal if second color is close
    if len(sorted_colors) > 1:
        second_color, second_prob = sorted_colors[1]
        if second_prob >= min_confidence - 0.05:  # Slightly lower threshold
            signals.append(Signal(
                "SECONDARY",
                f"BET_{second_color}",
                second_prob,
                f"Secondary signal: {second_color} at {second_prob:.2f}",
                probs=color_probs,
                sources=["Secondary"]
            ))
    
    return signals
