"""
Unit and Integration Tests for Feature 16: Web Dashboard TCA Integration & Heatmap Endpoints.
Verifies:
1. System status endpoint /api/status with TCA & Multi-Horizon capabilities
2. 240-candle extraction endpoint /api/candles
3. Augmented /api/analyze endpoint with multi-horizon predictions and non-zero telemetry
4. Dedicated /api/tca_heatmap and /api/tca/attention unified endpoints
5. Static index.html UI delivery with multi-horizon card and telemetry ribbon
6. Input validation and corner cases
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dashboard.backend import app, state, initialize_runtime


@pytest.fixture(scope="module")
def client():
    """Initializes runtime and creates FastAPI TestClient."""
    initialize_runtime()
    return TestClient(app)


class TestDashboardStatusEndpoint:
    """Verifies system status, loaded checkpoint, and TCA capabilities."""

    def test_status_endpoint_success(self, client):
        res = client.get("/api/status")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] in ["ready", "no_data"]
        assert data["lookback_candles"] == 240
        assert data["device"] == "cpu"
        if data["status"] == "ready":
            assert data["checkpoint_loaded"] is True
            assert data["tca_trained"] is True
            assert data["head_trained"] is True
            assert data["multi_horizon_active"] is True
            assert data["checkpoint_type"] in ["tca_phase2_finetuned", "tca_phase1_pretrained", "baseline_challenger"]


class TestCandlesEndpoint:
    """Verifies 240-candle extraction for TradingView chart."""

    def test_candles_endpoint_success(self, client):
        if state["total_bars"] < 240:
            pytest.skip("Not enough bars in environment for candles test")

        res = client.get("/api/candles?bar_idx=55000")
        assert res.status_code == 200
        data = res.json()
        assert data["target_bar_idx"] == 55000
        assert data["count"] == 240
        assert len(data["candles"]) == 240

        first_c = data["candles"][0]
        last_c = data["candles"][-1]
        for c in [first_c, last_c]:
            assert "time" in c
            assert "open" in c and c["open"] > 0
            assert "high" in c and c["high"] >= c["open"]
            assert "low" in c and c["low"] <= c["open"]
            assert "close" in c and c["close"] > 0
            assert "atr" in c and c["atr"] > 0


class TestAnalyzeEndpoint:
    """Verifies full forward inference, action recommendation, telemetry, and multi-horizon projections."""

    def test_analyze_endpoint_success(self, client):
        if state["total_bars"] < 240:
            pytest.skip("Not enough bars in environment for analyze test")

        res = client.post("/api/analyze", json={"bar_idx": 55000})
        assert res.status_code == 200
        data = res.json()

        # Decision & Action
        assert data["bar_idx"] == 55000
        assert data["action"] in ["BUY", "SELL", "HOLD"]
        assert data["action_code"] in [0, 1, 2]
        assert 0.0 <= data["confidence"] <= 1.0

        # Suggested Geometry
        assert "suggested_geometry" in data
        assert "sl_price" in data["suggested_geometry"]
        assert "tp_price" in data["suggested_geometry"]

        # Attention Telemetry (Verifies bug fix: recency_ratio and entropy must NOT be 0.0)
        attn = data["attention"]
        assert len(attn["mean_weights_240"]) == 240
        assert len(attn["head_weights"]) == 4
        assert len(attn["head_weights"][0]) == 240
        assert len(attn["top_5_candles"]) == 5
        assert attn["recency_ratio"] > 0.0, "recency_ratio must be positive (fixed key bug)"
        assert attn["entropy"] > 0.0, "entropy must be positive (fixed key bug)"
        assert len(attn["head_entropy"]) == 4
        assert len(attn["peak_indices"]) == 4

        # Multi-Horizon Projections
        assert "multi_horizon" in data
        mh = data["multi_horizon"]
        assert mh["horizons"] == [5, 15, 30]
        assert len(mh["pred_returns"]) == 3
        assert mh["consensus_direction"] in ["BULLISH", "BEARISH", "NEUTRAL"]
        assert "t+5" in mh["projections"]
        assert "t+15" in mh["projections"]
        assert "t+30" in mh["projections"]
        for h_key in ["t+5", "t+15", "t+30"]:
            p = mh["projections"][h_key]
            assert "atr_return" in p
            assert "price_delta" in p
            assert "target_price" in p
            assert p["signal"] in ["BULLISH", "BEARISH", "NEUTRAL"]


class TestTCAHeatmapUnifiedEndpoint:
    """Verifies dedicated /api/tca_heatmap and alias /api/tca/attention."""

    def test_tca_heatmap_payload(self, client):
        if state["total_bars"] < 240:
            pytest.skip("Not enough bars in environment for heatmap test")

        for endpoint in ["/api/tca_heatmap?bar_idx=55000", "/api/tca/attention?bar_idx=55000"]:
            res = client.get(endpoint)
            assert res.status_code == 200
            data = res.json()

            assert data["bar_idx"] == 55000
            assert "checkpoint" in data
            assert len(data["candles"]) == 240

            # 4 Attention heads of length 240
            attn = data["attention"]
            assert len(attn["heads"]) == 4
            for h in range(4):
                assert len(attn["heads"][h]) == 240
                assert abs(sum(attn["heads"][h]) - 1.0) < 1e-4

            # Mean weights of length 240
            assert len(attn["mean_weights_240"]) == 240
            assert abs(sum(attn["mean_weights_240"]) - 1.0) < 1e-4

            # Telemetry
            telemetry = attn["telemetry"]
            assert len(telemetry["entropy_per_head"]) == 4
            assert telemetry["mean_entropy"] > 0.0
            assert len(telemetry["peak_indices"]) == 4
            assert telemetry["recency_ratio"] > 0.0

            # Multi-horizon predictions
            mhp = data["multi_horizon_predictions"]
            assert mhp["horizons"] == [5, 15, 30]
            assert len(mhp["predicted_returns_atr"]) == 3
            assert mhp["consensus_direction"] in ["BULLISH", "BEARISH", "NEUTRAL"]


class TestDashboardFrontendAndValidation:
    """Verifies frontend HTML serving and endpoint validation."""

    def test_root_index_html_serving(self, client):
        res = client.get("/")
        assert res.status_code == 200
        html = res.text
        assert "TEMPORAL ATTENTION HEATMAP RIBBON" in html
        assert "TCA Multi-Horizon Projections" in html
        assert "telemetry-entropy" in html
        assert "telemetry-recency" in html

    def test_out_of_bounds_bar_index(self, client):
        res = client.get("/api/candles?bar_idx=99999999")
        assert res.status_code == 400

    def test_missing_body_analyze_fails(self, client):
        res = client.post("/api/analyze", json={})
        assert res.status_code == 422
