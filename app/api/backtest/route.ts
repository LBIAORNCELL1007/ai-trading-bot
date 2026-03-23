import { NextRequest, NextResponse } from 'next/server';
import { createBinanceClient } from '@/lib/binance-client';
import { Backtester } from '@/lib/backtester';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      symbol = 'BTCUSDT',
      timeframe = '1h',
      days = 30,
      initialBalance = 10000,
      riskPerTrade = 2,
      rewardMultiplier = 3,
      maxPositions = 5,
    } = body;

    const binanceClient = createBinanceClient(false);

    // Fetch historical data
    const endTime = Date.now();
    const startTime = endTime - days * 24 * 60 * 60 * 1000;

    const klines = await binanceClient.getKlines(
      symbol,
      timeframe,
      1000,
      startTime,
      endTime
    );

    if (klines.length === 0) {
      return NextResponse.json(
        { error: 'No historical data available' },
        { status: 400 }
      );
    }

    // Run backtest
    const backtester = new Backtester(initialBalance);
    const result = await backtester.backtest(symbol, klines, {
      riskPerTrade,
      rewardMultiplier,
      maxPositions,
    });

    const tradeStats = backtester.getTradeStats();
    const balanceHistory = backtester.getBalanceHistory();
    const returns = backtester.getReturns();

    return NextResponse.json({
      result,
      tradeStats,
      balanceHistory: balanceHistory.map((balance, index) => ({
        index,
        balance,
      })),
      returns: returns.map((ret, index) => ({
        index,
        return: ret,
      })),
    });
  } catch (error) {
    return NextResponse.json(
      { error: `Backtest failed: ${error}` },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbol = searchParams.get('symbol') || 'BTCUSDT';
  const timeframe = searchParams.get('timeframe') || '1h';
  const days = parseInt(searchParams.get('days') || '30');

  try {
    const binanceClient = createBinanceClient(false);

    const endTime = Date.now();
    const startTime = endTime - days * 24 * 60 * 60 * 1000;

    const klines = await binanceClient.getKlines(
      symbol,
      timeframe,
      1000,
      startTime,
      endTime
    );

    // Quick backtest with default settings
    const backtester = new Backtester(10000);
    const result = await backtester.backtest(symbol, klines, {
      riskPerTrade: 2,
      rewardMultiplier: 3,
      maxPositions: 5,
    });

    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: `Failed to fetch backtest data: ${error}` },
      { status: 500 }
    );
  }
}
