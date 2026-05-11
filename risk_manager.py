"""
Institutional Risk Controls for the Live Trading Bot.
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone
import json

import live_state as state

@dataclass
class RiskConfig:
    initial_paper_equity: float = 10000.0
    kelly_fraction: float = 0.5
    max_risk_per_trade_pct: float = 0.02
    max_daily_loss_pct: float = 0.06
    max_drawdown_pct: float = 0.15
    max_concurrent_positions: int = 5
    max_symbol_exposure_pct: float = 0.20
    var_confidence_level: float = 0.99
    max_var_pct: float = 0.05
    # Simplistic correlation tracking
    high_correlation_pairs: tuple = (("BTCUSDT", "ETHUSDT"), ("SOLUSDT", "AVAXUSDT"))

class RiskManager:
    def __init__(self, client, paper: bool, cfg: RiskConfig = None):
        self.client = client
        self.paper = paper
        self.cfg = cfg or RiskConfig()

    def _get_live_equity(self) -> float:
        if not self.client:
            raise ValueError("Live mode requires Binance client")
        try:
            acc = self.client.account()
            return float(acc['totalMarginBalance'])
        except Exception as e:
            raise ValueError(f"Failed to fetch live equity: {e}")

    def sync_risk_state(self) -> Optional[dict]:
        """Update peak equity and daily start equity in SQLite."""
        rs = state.get_risk_state()
        now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        if self.paper:
            if not rs:
                rs = {
                    "paper_equity": self.cfg.initial_paper_equity,
                    "peak_equity": self.cfg.initial_paper_equity, 
                    "daily_start_equity": self.cfg.initial_paper_equity, 
                    "last_trade_date": now_date
                }
                state.insert_risk_state(rs)
            current_equity = rs["paper_equity"]
        else:
            try:
                current_equity = self._get_live_equity()
            except Exception as e:
                state.log_event("risk_sync_error", level="ERROR", payload={"error": str(e)})
                return None
                
            if not rs:
                rs = {
                    "paper_equity": current_equity, 
                    "peak_equity": current_equity, 
                    "daily_start_equity": current_equity, 
                    "last_trade_date": now_date
                }
                state.insert_risk_state(rs)

        # Update peak and daily if needed
        updates = {}
        if current_equity > rs["peak_equity"]:
            updates["peak_equity"] = current_equity
            rs["peak_equity"] = current_equity
            
        if now_date != rs["last_trade_date"]:
            updates["daily_start_equity"] = current_equity
            updates["last_trade_date"] = now_date
            rs["daily_start_equity"] = current_equity
            rs["last_trade_date"] = now_date

        if not self.paper and abs(rs.get("paper_equity", 0) - current_equity) > 0.01:
             updates["paper_equity"] = current_equity
             rs["paper_equity"] = current_equity

        if updates:
            state.update_risk_state(updates)
            
        return rs

    def update_paper_equity(self, pnl_usdt: float, fee_paid: float = 0.0):
        if self.paper:
            rs = state.get_risk_state()
            if rs:
                new_eq = rs["paper_equity"] + pnl_usdt - fee_paid
                state.update_risk_state({"paper_equity": new_eq})

    def check_pre_trade_gates(
        self, 
        symbol: str, 
        price: float, 
        win_rate: float = 0.42, # Expected WR
        tp_pct: float = 0.02, 
        sl_pct: float = 0.02
    ) -> dict:
        """
        Runs institutional risk checks.
        Returns dict with 'passed': bool, 'qty': float, 'reason': str, 'risk_usdt': float
        """
        rs = self.sync_risk_state()
        if not rs:
            return {"passed": False, "qty": 0.0, "reason": "Missing or invalid risk state"}

        equity = rs["paper_equity"]
        if equity <= 0:
             return {"passed": False, "qty": 0.0, "reason": "Account equity <= 0"}

        # 1. Circuit Breaker (Max Daily Loss)
        daily_loss_pct = (rs["daily_start_equity"] - equity) / rs["daily_start_equity"] if rs["daily_start_equity"] > 0 else 0
        if daily_loss_pct >= self.cfg.max_daily_loss_pct:
            return {"passed": False, "qty": 0.0, "reason": f"Circuit breaker: Daily loss {daily_loss_pct*100:.2f}% >= {self.cfg.max_daily_loss_pct*100}%"}

        # 2. Max Drawdown Kill Switch
        dd_pct = (rs["peak_equity"] - equity) / rs["peak_equity"] if rs["peak_equity"] > 0 else 0
        if dd_pct >= self.cfg.max_drawdown_pct:
            return {"passed": False, "qty": 0.0, "reason": f"Kill switch: Max DD {dd_pct*100:.2f}% >= {self.cfg.max_drawdown_pct*100}%"}

        open_positions = state.list_open_positions()

        # 3. Max Concurrent Positions
        if len(open_positions) >= self.cfg.max_concurrent_positions:
            return {"passed": False, "qty": 0.0, "reason": f"Max concurrent positions reached ({self.cfg.max_concurrent_positions})"}

        # 4. Correlation Check
        open_symbols = {p['symbol'] for p in open_positions}
        for group in self.cfg.high_correlation_pairs:
            group_set = set(group)
            if symbol in group_set:
                overlap = group_set.intersection(open_symbols)
                if overlap:
                    return {"passed": False, "qty": 0.0, "reason": f"Correlation limit: already holding {overlap}"}

        # 5. Fractional Kelly Sizing
        r = tp_pct / sl_pct
        kelly_pct = win_rate - (1.0 - win_rate) / r
        if kelly_pct <= 0:
            # Fallback to minimal risk if edge is negligible
            kelly_pct = 0.01
            
        target_risk_pct = kelly_pct * self.cfg.kelly_fraction
        target_risk_pct = min(target_risk_pct, self.cfg.max_risk_per_trade_pct)

        risk_usdt = equity * target_risk_pct
        notional_size = risk_usdt / sl_pct
        
        # 6. Symbol Exposure Limit
        max_notional = equity * self.cfg.max_symbol_exposure_pct
        if notional_size > max_notional:
            notional_size = max_notional
            risk_usdt = notional_size * sl_pct

        # 7. VaR Check (Parametric VaR simplified: assume approx 1-day volatility = SL)
        # Using 5% portfolio VaR limit
        estimated_trade_var = notional_size * (sl_pct * 1.5) # heuristic 99% CI scaling
        if estimated_trade_var > (equity * self.cfg.max_var_pct):
            notional_size = (equity * self.cfg.max_var_pct) / (sl_pct * 1.5)
            risk_usdt = notional_size * sl_pct

        qty = notional_size / price

        return {
            "passed": True, 
            "qty": qty, 
            "reason": "Risk checks passed",
            "notional": notional_size,
            "risk_usdt": risk_usdt,
            "equity": equity
        }
