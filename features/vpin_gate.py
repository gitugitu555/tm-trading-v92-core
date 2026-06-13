class VPINGate:
    def __init__(self, window: int = 50, high_threshold: float = 0.65):
        self._window = window
        self._high_threshold = high_threshold
        self._buckets: list[float] = []

    def update(self, buy_volume: float, sell_volume: float) -> dict:
        total_vol = buy_volume + sell_volume
        if total_vol > 0:
            vpin_current = abs(buy_volume - sell_volume) / total_vol
        else:
            vpin_current = 0.0
            
        self._buckets.append(vpin_current)
        if len(self._buckets) > self._window:
            self._buckets.pop(0)
            
        if len(self._buckets) == self._window:
            vpin_rolling_mean = sum(self._buckets) / self._window
        else:
            vpin_rolling_mean = None
            
        toxicity_state = "LOW"
        if vpin_current > self._high_threshold:
            toxicity_state = "HIGH"
        elif vpin_current < 0.3:
            toxicity_state = "LOW"
        elif vpin_rolling_mean is not None:
            if vpin_current > vpin_rolling_mean and toxicity_state != "HIGH":
                toxicity_state = "RISING"
            elif vpin_current < vpin_rolling_mean and toxicity_state != "LOW":
                toxicity_state = "FALLING"
                
        return {
            "vpin_current": vpin_current,
            "vpin_rolling_mean": vpin_rolling_mean,
            "toxicity_state": toxicity_state,
            "should_block_branch_a": toxicity_state == "HIGH",
            "should_block_branch_b": toxicity_state == "HIGH"
        }
