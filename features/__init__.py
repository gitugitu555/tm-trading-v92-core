"""Feature package.

Keep package import side-effect free so focused tests can import helper modules
such as :mod:`features.v92_data_policy` without requiring every legacy engine
dependency to be installed.
"""

__all__: list[str] = []

try:  # Optional legacy engines.
    from .absorption import AbsorptionEngine
    from .cvd import CVDEngine
    from .delta import DeltaEngine
    from .footprint import FootprintEngine
    from .market_profile import MarketProfileEngine
    from .iceberg import IcebergDetector
    from .l2_imbalance import OrderBookImbalanceEngine
    from .large_prints import LargePrintDetector
    from .anti_patterns import AntiPatternEngine
    from .atr_context import ATRContextEngine
    from .mlofi import MLOFIEngine
    from .microprice import microprice
    from .queue_imbalance import QueueImbalanceEngine
    from .spoofing import SpoofingDetector
    from .trade_signing import TradeSigner, bvc_classify
    from .vpin import VPINEngine
    from .whale import WhalePressureEngine
except ModuleNotFoundError:
    pass
else:
    __all__ = [
        "AbsorptionEngine",
        "CVDEngine",
        "DeltaEngine",
        "FootprintEngine",
        "MarketProfileEngine",
        "IcebergDetector",
        "LargePrintDetector",
        "AntiPatternEngine",
        "ATRContextEngine",
        "MLOFIEngine",
        "OrderBookImbalanceEngine",
        "QueueImbalanceEngine",
        "SpoofingDetector",
        "TradeSigner",
        "VPINEngine",
        "WhalePressureEngine",
        "bvc_classify",
        "microprice",
    ]
