# Code Review — Order Execution Logic

**Scope**: [trading-engine.ts](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts), [binance-client.ts](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/binance-client.ts), [risk-management.ts](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/risk-management.ts), [backtester.ts](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/backtester.ts)

---

## 🔴 Critical — Will Lose Money

### 1. Entry uses LIMIT but assumes immediate fill

[executeBuy](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts#L180-L237) places a **LIMIT order** ([buyLimit](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/binance-client.ts#226-235)) but immediately marks the trade `OPEN` and registers a position — without waiting for the order to actually fill.

```typescript
// L211: places a LIMIT order
const order = await this.binanceClient.buyLimit(symbol, quantity, currentPrice);
// L233-234: immediately assumes it filled
this.riskManager.addPosition(position);
this.openTrades.push({ ...trade, status: 'OPEN' });
```

**Problem**: A limit order at `currentPrice` can easily sit unfilled if price moves away. The engine now thinks it has a position it doesn't have, and the exit logic will try to sell quantity it doesn't own.

> [!CAUTION]
> **Fix**: Either (a) use **MARKET orders** for entries on signal (like exits already do), or (b) poll `GET /v3/order` until `status === 'FILLED'` before registering the position, or (c) set up a user data stream WebSocket to listen for fill events.

### 2. Same issue on sell-side exits

[executeSell](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts#L242-L293) places a **LIMIT sell** but immediately records PnL and removes the position. If the limit doesn't fill, the position is orphaned on the exchange.

### 3. No order tracking at all

The return value from [placeOrder()](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/binance-client.ts#178-207) (which contains `orderId`, `status`, `executedQty`, `fills[]`) is logged then **thrown away**. There's no [Order](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/binance-client.ts#178-207) model, no order-ID storage, and no fill-price tracking. This means:

- You can't cancel stale unfilled orders
- You can't reconcile actual fills vs expected fills
- You can't handle partial fills

---

## 🟠 High — Correctness / Risk Gaps

### 4. Race condition: execution loop vs. exit loop

[checkPositionExits](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts#L298-L346) runs on **every WebSocket tick**, while [executeStrategy](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts#L157-L175) runs every **5 seconds** via `setInterval`. Both mutate `this.openTrades` concurrently. Since JS is single-threaded this is safe from data corruption, **but**:

- A sell signal fires in [executeStrategy](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts#154-176) for a position that [checkPositionExits](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts#295-347) already closed earlier in the same event-loop tick → double sell attempt
- No guard in [executeSell](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts#239-294) checks if the trade is still actually open at the moment of exchange call

> [!WARNING]
> Add an `if (openTrade.status !== 'OPEN') continue;` re-check inside the exit path, and use a locking flag on the trade object to prevent concurrent exit attempts.

### 5. Balance never synced with exchange

`RiskManager.accountBalance` is set once in the constructor and **never updated** from the exchange. After live trades, the local balance diverges from reality. [canOpenPosition()](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/risk-management.ts#227-247) and [calculatePositionSizeFixedRisk()](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/risk-management.ts#208-226) then compute wrong sizes.

> [!IMPORTANT]
> Periodically call [getAccountInfo()](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/binance-client.ts#112-115) and update `accountBalance` to the actual available balance.

### 6. Trailing stop logic is broken

[shouldClosePosition](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/risk-management.ts#L312-L317):

```typescript
if (position.trailingStop && position.side === 'LONG') {
  const trailPrice = currentPrice * (1 - position.trailingStop / 100);
  if (currentPrice <= trailPrice) {  // ❌ always false
```

`trailPrice` is always **below** `currentPrice` (it's `currentPrice * 0.97` for a 3% trail), so `currentPrice <= trailPrice` is **always false**. A trailing stop needs to track the **highest price since entry** and then check if price has dropped from that peak.

---

## 🟡 Medium — Robustness

### 7. No MIN_NOTIONAL check

Binance rejects orders below a minimum notional value (e.g., $5 for most pairs). [placeOrder](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/binance-client.ts#L178-L206) rounds quantity and price but doesn't validate `quantity * price >= minNotional`. Small-balance accounts will get silent rejections.

### 8. No retry on transient API failures

[executeBuy](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts#L209-L221) catches errors and returns, but doesn't distinguish between:
- `-1021 INVALID_TIMESTAMP` (clock drift — fixable with retry)
- `-2010 INSUFFICIENT_BALANCE` (real error — stop)
- `-1015 TOO_MANY_ORDERS` (rate limit — retry after delay)

### 9. [executeSell](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts#239-294) closes only the first open trade

[L252](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts#L252): `const trade = openBuyTrades[0]` — if you have multiple positions on the same symbol, sell signals always close the oldest one. This is probably intentional (FIFO), but worth documenting because it means newer profitable positions can't be exited on a sell signal while the oldest loser sits.

### 10. Exchange info cache never expires

[getExchangeInfo](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/binance-client.ts#L268-L295) caches forever. If Binance changes lot size / tick size (they do, especially for new pairs), the bot will send invalid quantities until restarted.

### 11. `stopPrice` not rounded

[placeOrder L201-203](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/binance-client.ts#L201-L203) — `stopPrice` is passed as-is without [roundPrice()](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/binance-client.ts#308-312). Binance will reject stop orders with invalid tick size.

---

## 🔵 Low — Code Quality / Maintainability

| # | Issue | Location |
|---|---|---|
| 12 | [Trade](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts#18-33) and [Position](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/risk-management.ts#1-11) are nearly identical models with no shared base — syncing them is error-prone | [trading-engine.ts](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts) L18-32 vs [risk-management.ts](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/risk-management.ts) L1-10 |
| 13 | `tradeHistory: any[]` in RiskManager — no type safety for trade records | [risk-management.ts](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/risk-management.ts) L29 |
| 14 | WebSocket connections stored in `any[]` — no type for lifecycle mgmt | [trading-engine.ts](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts) L45 |
| 15 | `priceHistory` grows unboundedly — memory leak on long runs | [trading-engine.ts](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts) L43 |
| 16 | `closedTrades` also grows unboundedly — should cap or archive | [trading-engine.ts](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/trading-engine.ts) L39 |

---

## Summary — Priority Fix Order

| Priority | Issue | Effort |
|----------|-------|--------|
| 🔴 P0 | Use MARKET orders for entry **or** implement fill tracking | Small (use market) / Large (fill tracking) |
| 🔴 P0 | Store `orderId` and check fill status before registering positions | Medium |
| 🟠 P1 | Fix trailing stop (track high-water mark) | Small |
| 🟠 P1 | Guard against double-exit race condition | Small |
| 🟠 P1 | Sync `accountBalance` with exchange periodically | Small |
| 🟡 P2 | Add MIN_NOTIONAL pre-validation | Small |
| 🟡 P2 | Round `stopPrice` in [placeOrder](file:///c:/Users/91892/Downloads/ai-trading-bot/lib/binance-client.ts#178-207) | Small |
| 🟡 P2 | Add exchange info cache TTL | Small |
| 🟡 P2 | Classify + retry transient API errors | Medium |
