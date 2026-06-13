import polars as pl

def add_regime_labels(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        ((pl.col("close") - pl.col("open")).abs() / pl.col("volume_delta").abs().clip(lower_bound=1e-9)).alias("delta_efficiency")
    ])

    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(1) - 1).rolling_std(window_size=500).shift(1).alias("rv_1d"),
        pl.col("volume").rolling_quantile(quantile=0.80, interpolation="nearest", window_size=3500).shift(1).alias("vol_80th_pct"),
        pl.col("delta_efficiency").rolling_median(window_size=200).shift(1).alias("de_median"),
        pl.col("low").rolling_min(window_size=500).shift(1).alias("rolling_min_low"),
        (pl.col("high") - pl.col("low")).rolling_mean(window_size=7000).shift(1).alias("rolling_mean_range")
    ])

    df = df.with_columns([
        pl.col("rv_1d").rolling_quantile(quantile=0.15, interpolation="nearest", window_size=15000).shift(1).alias("rv_15th_pct"),
        ((pl.col("close") - pl.col("rolling_min_low")) / pl.col("rolling_mean_range")).alias("adr_stretch")
    ])

    df = df.with_columns([
        pl.when(pl.col("rv_1d") < pl.col("rv_15th_pct"))
        .then(pl.lit("TREND_BUILDUP"))
        .when(
            (pl.col("volume") > pl.col("vol_80th_pct")) &
            (pl.col("body_size") < (0.25 * pl.col("bar_range"))) &
            (pl.col("delta_efficiency") < pl.col("de_median"))
        )
        .then(pl.lit("ABSORPTION"))
        .when((pl.col("adr_stretch") > 0.85) | (pl.col("adr_stretch") < 0.15))
        .then(pl.lit("EXHAUSTED"))
        .otherwise(pl.lit("NOISE"))
        .alias("regime")
    ])

    return df.drop(["rv_1d", "vol_80th_pct", "rolling_min_low", "rolling_mean_range", "rv_15th_pct", "adr_stretch"])
