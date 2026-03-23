import { NextRequest, NextResponse } from 'next/server';
import { createBinanceClient } from '@/lib/binance-client';
import { TradingEngine, TradeExecutionConfig } from '@/lib/trading-engine';
import { Backtester } from '@/lib/backtester';

// ⚠️  Global mutable state — acceptable for single-instance dev/paper mode.
// For production multi-instance deployment, replace with Redis or DB-backed state.
let tradingEngine: TradingEngine | null = null;
let engineConfig: TradeExecutionConfig | null = null;
let isStarting = false; // Prevents concurrent start race conditions

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, config, symbol, timeframe } = body;

    if (action === 'start') {
      if (tradingEngine && tradingEngine.isEngineRunning()) {
        return NextResponse.json({ error: 'Trading engine already running' }, { status: 400 });
      }
      if (isStarting) {
        return NextResponse.json({ error: 'Trading engine is starting up, please wait' }, { status: 429 });
      }
      isStarting = true;

      const binanceClient = createBinanceClient(config?.testnet || false);
      const defaultConfig: TradeExecutionConfig = {
        symbol: symbol || 'BTCUSDT',
        strategy: 'MULTI_SIGNAL',
        timeframe: timeframe || '1h',
        riskPerTrade: config?.riskPerTrade || 2,
        rewardMultiplier: config?.rewardMultiplier || 3,
        maxPositions: config?.maxPositions || 5,
        paperTrading: config?.paperTrading !== false,
        gridCount: config?.gridCount || 10,
        gridPercentage: config?.gridPercentage || 0.02,
      };

      try {
        const accountInfo = await binanceClient.getAccountInfo();
        const balance = parseFloat(
          accountInfo.balances.find((b: any) => b.asset === 'USDT')?.free || '0'
        );

        tradingEngine = new TradingEngine(binanceClient, balance, defaultConfig);
        engineConfig = defaultConfig;
        await tradingEngine.start();
        isStarting = false;

        return NextResponse.json({
          status: 'started',
          config: defaultConfig,
          balance,
        });
      } catch (error) {
        isStarting = false;
        return NextResponse.json(
          { error: `Failed to start trading engine: ${error}` },
          { status: 500 }
        );
      }
    }

    if (action === 'stop') {
      if (tradingEngine) {
        await tradingEngine.stop();
        tradingEngine = null;
        engineConfig = null;
        return NextResponse.json({ status: 'stopped' });
      }
      return NextResponse.json({ error: 'No trading engine running' }, { status: 400 });
    }

    if (action === 'status') {
      if (tradingEngine) {
        return NextResponse.json({
          running: tradingEngine.isEngineRunning(),
          config: engineConfig,
          openTrades: tradingEngine.getOpenTrades(),
          closedTrades: tradingEngine.getClosedTrades(),
          portfolio: tradingEngine.getPortfolioMetrics(),
          signals: tradingEngine.getLatestSignals(),
          price: tradingEngine.getCurrentPrice(),
        });
      }
      return NextResponse.json({ running: false });
    }

    return NextResponse.json({ error: 'Unknown action' }, { status: 400 });
  } catch (error) {
    return NextResponse.json(
      { error: `API Error: ${error}` },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const action = searchParams.get('action');

  if (action === 'status') {
    if (tradingEngine) {
      return NextResponse.json({
        running: tradingEngine.isEngineRunning(),
        config: engineConfig,
        openTrades: tradingEngine.getOpenTrades(),
        closedTrades: tradingEngine.getClosedTrades(),
        portfolio: tradingEngine.getPortfolioMetrics(),
        signals: tradingEngine.getLatestSignals(),
        price: tradingEngine.getCurrentPrice(),
      });
    }
    return NextResponse.json({ running: false });
  }

  return NextResponse.json({ error: 'Unknown action' }, { status: 400 });
}
