# HFT Data Manifest Schema (Binance Futures)

## Overview
This document outlines the raw schema of the institutional HFT data stored on the Seagate drive. The data consists of highly-compressed Zstandard Parquet files (`.parquet.zst`) representing continuous tick-level orderbook streams.

## Orderbook Schema (`orderbook` data type)
These files contain the raw WebSocket `depthUpdate` payload from Binance Futures. 

*Note: Price and Quantity are stored as strings (objects) to prevent floating-point precision loss during capture, and must be cast to `float64` during processing.*

| Column | Type | Description |
| :--- | :--- | :--- |
| `received_time` | `int64` | Local nanosecond timestamp of when the packet was received by the capture server (crucial for latency/slippage simulation). |
| `event_time` | `int64` | Exchange millisecond timestamp (`E` field). |
| `transaction_time` | `int64` | Exchange millisecond transaction time (`T` field). |
| `symbol` | `object` | Ticker symbol (e.g. `BTCUSDT`). |
| `event_type` | `object` | Type of payload (e.g. `update`, `snapshot`). |
| `first_update_id` | `int64` | `U` field in Binance docs. |
| `final_update_id` | `int64` | `u` field in Binance docs. |
| `prev_final_update_id` | `int64` | `pu` field in Binance docs (used to ensure no packets were dropped). |
| `last_update_id` | `float64` | Used for snapshots. |
| `side` | `object` | `bid` or `ask`. |
| `price` | `object` | Price level updated (String). |
| `quantity` | `object` | New quantity at the price level (String). A quantity of `0` means the price level was removed. |
| `order_count` | `float64` | (Optional) Number of orders if available. |

## Processing Notes
Because this is a continuous delta stream (`update` events), generating top-of-book (BBO) or depth features (like OFI) requires maintaining a local order book state machine in memory that applies these diffs sequentially.
