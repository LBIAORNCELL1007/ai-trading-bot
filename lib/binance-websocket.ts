// Binance WebSocket service for real-time price data
// Free to use - no API key required for public market data

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
