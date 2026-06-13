def check_branch_a_anti_patterns(row: dict, side: str = "LONG") -> tuple[bool, list[str]]:
    reasons = []
    
    close_quality = row.get("close_quality", 0.5)
    volume_delta = row.get("volume_delta", 0)
    
    if side == "LONG":
        if close_quality < 0.50:
            reasons.append("WEAK_CLOSE_QUALITY")
        if 0 < volume_delta < 50:
            reasons.append("THIN_DELTA_BREAKOUT")
    else:  # SHORT
        if close_quality > 0.50:
            reasons.append("WEAK_CLOSE_QUALITY")
        if -50 < volume_delta < 0:
            reasons.append("THIN_DELTA_BREAKOUT")
            
    if row.get("delta_efficiency", 1.0) < 0.0002:
        reasons.append("ABSORBED_BREAKOUT")
        
    return len(reasons) > 0, reasons

def check_branch_c_anti_patterns(row: dict) -> tuple[bool, list[str]]:
    reasons = []
    
    if row.get("volume_delta", 0) > 100:
        reasons.append("STRONG_BUYER_FLOW_AT_HIGH")
        
    if row.get("close_quality", 0.0) > 0.70:
        reasons.append("NO_REJECTION_CLOSE")
        
    return len(reasons) > 0, reasons
