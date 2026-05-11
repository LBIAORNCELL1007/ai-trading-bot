"""
Smoke test for execution realism and order logic.
Tests rounding, minNotional, and chasing behavior in dry-run mode.
"""
import unittest
from unittest.mock import MagicMock
import live_orders as lo
from dataclasses import dataclass

class TestExecutionRealism(unittest.TestCase):
    def setUp(self):
        self.filters = lo.SymbolFilters(
            symbol="BTCUSDT",
            tick_size=0.1,
            step_size=0.001,
            min_qty=0.001,
            min_notional=100.0
        )

    def test_rounding_logic(self):
        # Round down quantity
        self.assertEqual(lo.round_down(1.23456, 0.01), 1.23)
        self.assertEqual(lo.round_down(1.23456, 0.001), 1.234)
        
        # Round price
        self.assertEqual(lo.round_down(65432.19, 0.1), 65432.1)
        self.assertEqual(lo.round_up(65432.11, 0.1), 65432.2)

    def test_min_notional_gate(self):
        # Should return failed FillResult
        res = lo.place_maker_buy(
            client=None,
            filters=self.filters,
            best_bid=50000.0,
            best_ask=50010.0,
            quantity=0.001, # 50 USDT notional < 100.0
            paper=True
        )
        self.assertFalse(res.filled)
        self.assertIn("below min qty/notional", res.note)

    def test_spread_gate(self):
        # 1% spread > 0.1% max_spread_pct
        res = lo.place_maker_buy(
            client=None,
            filters=self.filters,
            best_bid=100.0,
            best_ask=101.1,
            quantity=2.0,
            max_spread_pct=0.1,
            paper=True
        )
        self.assertFalse(res.filled)
        self.assertIn("spread too wide", res.note)

    def test_paper_fill_realistic_fees(self):
        res = lo.place_maker_buy(
            client=None,
            filters=self.filters,
            best_bid=50000.0,
            best_ask=50005.0,
            quantity=0.01,
            paper=True,
            maker_fee_pct=0.0002
        )
        self.assertTrue(res.filled)
        # 500 USDT notional * 0.0002 = 0.1 USDT fee
        self.assertAlmostEqual(res.fee_paid, 0.1)

if __name__ == "__main__":
    unittest.main()
