// Binance WebSocket service for real-time price data
// Free to use - no API key required for public market data

import type { Time } from 'lightweight-charts'

const BINANCE_WS_BASE = 'wss://stream.binance.com:9443/ws'

export interface BinanceKline {
    time: number
    open: number
    high: number
    low: number
    close: number
    volume: number
    isClosed: boolean
}

export interface BinanceTicker {
    symbol: string
    price: number
    priceChange: number
    priceChangePercent: number
    volume: number
}

export interface PrecomputedMAs {
    ma7: { time: Time; value: number }[]
    ma25: { time: Time; value: number }[]
    ma99: { time: Time; value: number }[]
}

type KlineCallback = (kline: BinanceKline) => void
type TickerCallback = (ticker: BinanceTicker) => void
type PriceCallback = (price: number) => void

class BinanceWebSocket {
    private connections: Map<string, WebSocket> = new Map()
    private reconnectTimeouts: Map<string, NodeJS.Timeout> = new Map()

    // Subscribe to real-time kline/candlestick data
    subscribeKline(
        symbol: string,
        interval: string,
        callback: KlineCallback
    ): () => void {
        const streamName = `${symbol.toLowerCase()}@kline_${interval}`
        const wsUrl = `${BINANCE_WS_BASE}/${streamName}`

        const connect = () => {
            const ws = new WebSocket(wsUrl)

            ws.onopen = () => {
                console.log(`[Binance WS] Connected to ${streamName}`)
            }

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data)
                    if (data.k) {
                        const kline: BinanceKline = {
                            time: data.k.t,
                            open: parseFloat(data.k.o),
                            high: parseFloat(data.k.h),
                            low: parseFloat(data.k.l),
                            close: parseFloat(data.k.c),
                            volume: parseFloat(data.k.v),
                            isClosed: data.k.x,
                        }
                        callback(kline)
                    }
                } catch (error) {
                    console.error('[Binance WS] Parse error:', error)
                }
            }

            ws.onerror = (event: Event) => {
                // Silence generic "error" events which contain no useful info
                // These often occur on network disconnects before onclose
                const errorAny = event as any;
                if (errorAny.message && errorAny.message.length > 0) {
                    console.error(`[Binance WS] Error on ${streamName}:`, errorAny.message);
                }
            }

            ws.onclose = () => {
                console.log(`[Binance WS] Disconnected from ${streamName}`)
                // Reconnect after 3 seconds
                const timeout = setTimeout(() => {
                    if (this.connections.has(streamName)) {
                        connect()
                    }
                }, 3000)
                this.reconnectTimeouts.set(streamName, timeout)
            }

            this.connections.set(streamName, ws)
        }

        connect()

        // Return unsubscribe function
        return () => {
            const ws = this.connections.get(streamName)
            if (ws) {
                ws.close()
                this.connections.delete(streamName)
            }
            const timeout = this.reconnectTimeouts.get(streamName)
            if (timeout) {
                clearTimeout(timeout)
                this.reconnectTimeouts.delete(streamName)
            }
        }
    }

    // Subscribe to real-time price updates for a single symbol
    subscribePrice(symbol: string, callback: PriceCallback): () => void {
        const streamName = `${symbol.toLowerCase()}@trade`
        const wsUrl = `${BINANCE_WS_BASE}/${streamName}`

        const connect = () => {
            const ws = new WebSocket(wsUrl)

            ws.onopen = () => {
                console.log(`[Binance WS] Connected to ${streamName}`)
            }

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data)
                    if (data.p) {
                        callback(parseFloat(data.p))
                    }
                } catch (error) {
                    console.error('[Binance WS] Parse error:', error)
                }
            }

            ws.onerror = (error) => {
                console.error(`[Binance WS] Error on ${streamName}:`, error)
            }

            ws.onclose = () => {
                console.log(`[Binance WS] Disconnected from ${streamName}`)
                const timeout = setTimeout(() => {
                    if (this.connections.has(streamName)) {
                        connect()
                    }
                }, 3000)
                this.reconnectTimeouts.set(streamName, timeout)
            }

            this.connections.set(streamName, ws)
        }

        connect()

        return () => {
            const ws = this.connections.get(streamName)
            if (ws) {
                ws.close()
                this.connections.delete(streamName)
            }
            const timeout = this.reconnectTimeouts.get(streamName)
            if (timeout) {
                clearTimeout(timeout)
                this.reconnectTimeouts.delete(streamName)
            }
        }
    }

    // Subscribe to multiple tickers at once (for coins list)
    subscribeTickers(symbols: string[], callback: (tickers: Map<string, BinanceTicker>) => void): () => void {
        const streams = symbols.map(s => `${s.toLowerCase()}@ticker`).join('/')
        const wsUrl = `${BINANCE_WS_BASE}/stream?streams=${streams}`
        const tickers = new Map<string, BinanceTicker>()

        const connect = () => {
            const ws = new WebSocket(wsUrl)

            ws.onopen = () => {
                console.log(`[Binance WS] Connected to multi-ticker stream`)
            }

            ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data)
                    const data = message.data
                    if (data && data.s) {
                        const ticker: BinanceTicker = {
                            symbol: data.s,
                            price: parseFloat(data.c),
                            priceChange: parseFloat(data.p),
                            priceChangePercent: parseFloat(data.P),
                            volume: parseFloat(data.v),
                        }
                        tickers.set(data.s, ticker)
                        callback(new Map(tickers))
                    }
                } catch (error) {
                    console.error('[Binance WS] Parse error:', error)
                }
            }

            ws.onerror = (error) => {
                console.error(`[Binance WS] Error on multi-ticker:`, error)
            }

            ws.onclose = () => {
                console.log(`[Binance WS] Disconnected from multi-ticker`)
                const timeout = setTimeout(() => {
                    if (this.connections.has('multi-ticker')) {
                        connect()
                    }
                }, 3000)
                this.reconnectTimeouts.set('multi-ticker', timeout)
            }

            this.connections.set('multi-ticker', ws)
        }

        connect()

        return () => {
            const ws = this.connections.get('multi-ticker')
            if (ws) {
                ws.close()
                this.connections.delete('multi-ticker')
            }
            const timeout = this.reconnectTimeouts.get('multi-ticker')
            if (timeout) {
                clearTimeout(timeout)
                this.reconnectTimeouts.delete('multi-ticker')
            }
        }
    }

    // Close all connections
    closeAll() {
        this.connections.forEach((ws, key) => {
            ws.close()
        })
        this.connections.clear()
        this.reconnectTimeouts.forEach((timeout) => {
            clearTimeout(timeout)
        })
        this.reconnectTimeouts.clear()
    }
}

// Singleton instance
export const binanceWS = new BinanceWebSocket()

// Helper to get historical klines from REST API
export async function getHistoricalKlines(
    symbol: string,
    interval: string,
    limit: number = 100
): Promise<BinanceKline[]> {
    try {
        const response = await fetch(
            `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`
        )

        if (!response.ok) {
            throw new Error(`Binance API error: ${response.status}`)
        }

        const data = await response.json()

        return data.map((k: any[]) => ({
            time: k[0],
            open: parseFloat(k[1]),
            high: parseFloat(k[2]),
            low: parseFloat(k[3]),
            close: parseFloat(k[4]),
            volume: parseFloat(k[5]),
            isClosed: true,
        }))
    } catch (error) {
        console.warn(`[Binance API] Failed to fetch historical klines for ${symbol}. Falling back to mock data.`)
        // Generate mock data so the UI doesn't visually break for non-Binance pairs
        const mockKlines: BinanceKline[] = [];
        const now = Date.now();
        let lastClose = 100;
        
        // Interval string parsing to ms
        let intervalMs = 60000; // 1m default
        if (interval.endsWith('m')) intervalMs = parseInt(interval) * 60000;
        else if (interval.endsWith('h')) intervalMs = parseInt(interval) * 3600000;
        else if (interval.endsWith('d')) intervalMs = parseInt(interval) * 86400000;

        for (let i = limit; i > 0; i--) {
            const time = now - (i * intervalMs);
            const open = lastClose;
            const close = open * (1 + (Math.random() - 0.48) * 0.02);
            const high = Math.max(open, close) * (1 + Math.random() * 0.01);
            const low = Math.min(open, close) * (1 - Math.random() * 0.01);
            const volume = Math.random() * 1000;
            
            mockKlines.push({
                time, open, high, low, close, volume, isClosed: true
            });
            lastClose = close;
        }
        return mockKlines;
    }
}

// Fetch full history (all available) from CoinGecko API route and convert to OHLC-like candles
export async function getCoingeckoHistoryFull(
    coinId: string,
    days: string = 'max'
): Promise<{ klines: BinanceKline[]; mas: PrecomputedMAs }> {
    let points: Array<{ timestamp: string | number; price: number }> = []

    // 1) Try app API route first
    try {
        const response = await fetch(`/api/coins/${coinId}/history?days=${days}`)
        if (response.ok) {
            const json = await response.json()
            points = (json?.data || []) as Array<{ timestamp: string | number; price: number }>
        }
    } catch {
        // continue to fallback
    }

    // 2) Fallback direct CoinGecko public endpoint (no key path)
    if (!points.length) {
        const fallback = await fetch(
            `https://api.coingecko.com/api/v3/coins/${coinId}/market_chart?vs_currency=usd&days=${days}`
        )
        if (fallback.ok) {
            const data = await fallback.json()
            points = ((data?.prices || []) as [number, number][]).map((p) => ({
                timestamp: p[0],
                price: p[1],
            }))
        }
    }

    // 3) Final resilience fallback to Binance history if CoinGecko blocks (401/429/etc)
    if (!points.length) {
        const symbolMap: Record<string, string> = {
            bitcoin: 'BTCUSDT',
            ethereum: 'ETHUSDT',
            binancecoin: 'BNBUSDT',
            cardano: 'ADAUSDT',
            solana: 'SOLUSDT',
            ripple: 'XRPUSDT',
            dogecoin: 'DOGEUSDT',
            'matic-network': 'MATICUSDT',
            polkadot: 'DOTUSDT',
            litecoin: 'LTCUSDT',
            'avalanche-2': 'AVAXUSDT',
            chainlink: 'LINKUSDT',
            uniswap: 'UNIUSDT',
            cosmos: 'ATOMUSDT',
            'ethereum-classic': 'ETCUSDT',
        }
        const symbol = symbolMap[coinId] || 'BTCUSDT'
        const binanceKlines = await getHistoricalKlines(symbol, '1d', 1000)
        points = binanceKlines.map((k) => ({
            timestamp: k.time,
            price: k.close,
        }))
    }

    const klines: BinanceKline[] = points.map((p, idx) => {
        const time = typeof p.timestamp === 'number' ? p.timestamp : new Date(p.timestamp).getTime()
        const price = Number(p.price) || 0
        const prev = idx > 0 ? Number(points[idx - 1].price) || price : price
        const drift = Math.abs(price - prev)
        const high = Math.max(price, prev) + drift * 0.15
        const low = Math.max(0, Math.min(price, prev) - drift * 0.15)
        return {
            time,
            open: prev || price,
            high: high || price,
            low: low || price,
            close: price,
            volume: Math.max(1, drift * 1000), // synthetic volume
            isClosed: true,
        }
    })

    const buildMA = (input: BinanceKline[], period: number) => {
        const out: { time: Time; value: number }[] = []
        let sum = 0
        for (let i = 0; i < input.length; i++) {
            sum += input[i].close
            if (i >= period) sum -= input[i - period].close
            if (i >= period - 1) {
                out.push({
                    time: (input[i].time / 1000) as Time,
                    value: sum / period,
                })
            }
        }
        return out
    }

    return {
        klines,
        mas: {
            ma7: buildMA(klines, 7),
            ma25: buildMA(klines, 25),
            ma99: buildMA(klines, 99),
        },
    }
}

// Get current price from Binance REST API
export async function getCurrentPrice(symbol: string): Promise<number> {
    try {
        const response = await fetch(
            `https://api.binance.com/api/v3/ticker/price?symbol=${symbol}`
        )

        if (!response.ok) {
            throw new Error(`Binance API error: ${response.status}`)
        }

        const data = await response.json()
        return parseFloat(data.price)
    } catch (error) {
        console.error('Failed to fetch price:', error)
        return 0
    }
}
