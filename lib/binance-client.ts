import crypto from 'crypto';

interface BinanceConfig {
  apiKey: string;
  apiSecret: string;
  baseUrl?: string;
  testnet?: boolean;
}

interface OrderParams {
  symbol: string;
  side: 'BUY' | 'SELL';
  type: 'LIMIT' | 'MARKET' | 'STOP_LOSS' | 'TAKE_PROFIT';
  quantity: number;
  price?: number;
  stopPrice?: number;
  timeInForce?: 'GTC' | 'IOC' | 'FOK';
}

interface TickerData {
  symbol: string;
  price: number;
  time: number;
}

interface KlineData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export class BinanceClient {
  private apiKey: string;
  private apiSecret: string;
  private baseUrl: string;
  private websocketUrl: string;

  constructor(config: BinanceConfig) {
    this.apiKey = config.apiKey;
    this.apiSecret = config.apiSecret;
    
    if (config.testnet) {
      this.baseUrl = 'https://testnet.binance.vision/api';
      this.websocketUrl = 'wss://stream.testnet.binance.vision:9443/ws';
    } else {
      this.baseUrl = 'https://api.binance.com/api';
      this.websocketUrl = 'wss://stream.binance.com:9443/ws';
    }
  }

  private generateSignature(queryString: string): string {
    return crypto
      .createHmac('sha256', this.apiSecret)
      .update(queryString)
      .digest('hex');
  }

  private async makeRequest(
    endpoint: string,
    params?: Record<string, any>,
    method: 'GET' | 'POST' | 'DELETE' = 'GET',
    signed: boolean = false
  ): Promise<any> {
    let url = `${this.baseUrl}${endpoint}`;
    let body: string | undefined = undefined;

    if (params) {
      const queryString = new URLSearchParams(params).toString();
      if (signed) {
        const timestamp = Date.now();
        const signParams = { ...params, timestamp: timestamp.toString() };
        const signQueryString = new URLSearchParams(signParams).toString();
        const signature = this.generateSignature(signQueryString);
        if (method === 'GET' || method === 'DELETE') {
          url += `?${signQueryString}&signature=${signature}`;
        } else {
          // POST: send signed params in body
          body = `${signQueryString}&signature=${signature}`;
        }
      } else if (method === 'GET') {
        url += `?${queryString}`;
      } else {
        body = queryString;
      }
    }

    const headers: Record<string, string> = {
      'X-MBX-APIKEY': this.apiKey,
    };

    if (method !== 'GET') {
      headers['Content-Type'] = 'application/x-www-form-urlencoded';
    }

    const response = await fetch(url, {
      method,
      headers,
      body,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`Binance API Error: ${error.msg}`);
    }

    return response.json();
  }

  async getAccountInfo() {
    return this.makeRequest('/v3/account', {}, 'GET', true);
  }

  async getBalance(asset?: string) {
    const account = await this.getAccountInfo();
    if (asset) {
      return account.balances.find((b: any) => b.asset === asset);
    }
    return account.balances;
  }

  async getKlines(
    symbol: string,
    interval: string,
    limit: number = 100,
    startTime?: number,
    endTime?: number
  ): Promise<KlineData[]> {
    const params: any = {
      symbol,
      interval,
      limit,
    };

    if (startTime) params.startTime = startTime;
    if (endTime) params.endTime = endTime;

    const klines = await this.makeRequest('/v3/klines', params);
    return klines.map((k: any) => ({
      time: k[0],
      open: parseFloat(k[1]),
      high: parseFloat(k[2]),
      low: parseFloat(k[3]),
      close: parseFloat(k[4]),
      volume: parseFloat(k[5]),
    }));
  }

  async getTicker(symbol: string): Promise<TickerData> {
    const ticker = await this.makeRequest('/v3/ticker/24hr', { symbol });
    return {
      symbol: ticker.symbol,
      price: parseFloat(ticker.lastPrice),
      time: ticker.time,
    };
  }

  async getAllTickers(): Promise<TickerData[]> {
    const tickers = await this.makeRequest('/v3/ticker/24hr');
    return tickers.map((t: any) => ({
      symbol: t.symbol,
      price: parseFloat(t.lastPrice),
      time: t.time,
    }));
  }

  async getOpenOrders(symbol?: string) {
    const params = symbol ? { symbol } : {};
    return this.makeRequest('/v3/openOrders', params, 'GET', true);
  }

  async getOrderHistory(symbol: string, limit: number = 100) {
    return this.makeRequest('/v3/allOrders', { symbol, limit }, 'GET', true);
  }

  async placeOrder(params: OrderParams) {
    // Round quantity and price to exchange-allowed precision
    const roundedQty = await this.roundQuantity(params.symbol, params.quantity);
    const roundedPrice = params.price
      ? await this.roundPrice(params.symbol, params.price)
      : undefined;

    const orderParams: Record<string, string> = {
      symbol: params.symbol,
      side: params.side,
      type: params.type,
      quantity: roundedQty.toString(),
    };

    // Only include timeInForce for LIMIT-type orders (MARKET rejects it)
    if (params.type !== 'MARKET') {
      orderParams.timeInForce = params.timeInForce || 'GTC';
    }

    if (roundedPrice !== undefined) {
      orderParams.price = roundedPrice.toString();
    }

    if (params.stopPrice) {
      orderParams.stopPrice = params.stopPrice.toString();
    }

    return this.makeRequest('/v3/order', orderParams, 'POST', true);
  }

  async buyMarket(symbol: string, quantity: number) {
    return this.placeOrder({
      symbol,
      side: 'BUY',
      type: 'MARKET',
      quantity,
    });
  }

  async sellMarket(symbol: string, quantity: number) {
    return this.placeOrder({
      symbol,
      side: 'SELL',
      type: 'MARKET',
      quantity,
    });
  }

  async buyLimit(symbol: string, quantity: number, price: number) {
    return this.placeOrder({
      symbol,
      side: 'BUY',
      type: 'LIMIT',
      quantity,
      price,
    });
  }

  async sellLimit(symbol: string, quantity: number, price: number) {
    return this.placeOrder({
      symbol,
      side: 'SELL',
      type: 'LIMIT',
      quantity,
      price,
    });
  }

  async cancelOrder(symbol: string, orderId: number) {
    return this.makeRequest(
      '/v3/order',
      { symbol, orderId },
      'DELETE',
      true
    );
  }

  async getServerTime() {
    return this.makeRequest('/v3/time', {});
  }

  // ========================================================================
  // LOT SIZE & TICK SIZE ROUNDING
  // ========================================================================

  private exchangeInfoCache: Record<string, { lotStepSize: number; tickSize: number }> = {};

  /**
   * Fetch and cache exchange info (LOT_SIZE and PRICE_FILTER) for a symbol
   */
  async getExchangeInfo(symbol: string): Promise<{ lotStepSize: number; tickSize: number }> {
    if (this.exchangeInfoCache[symbol]) {
      return this.exchangeInfoCache[symbol];
    }

    try {
      const response = await fetch(`${this.baseUrl}/v3/exchangeInfo?symbol=${symbol}`);
      if (!response.ok) throw new Error(`Exchange info error: ${response.status}`);
      const data = await response.json();
      const symbolInfo = data.symbols?.[0];

      let lotStepSize = 0.00001;
      let tickSize = 0.01;

      if (symbolInfo?.filters) {
        const lotFilter = symbolInfo.filters.find((f: any) => f.filterType === 'LOT_SIZE');
        const priceFilter = symbolInfo.filters.find((f: any) => f.filterType === 'PRICE_FILTER');
        if (lotFilter) lotStepSize = parseFloat(lotFilter.stepSize);
        if (priceFilter) tickSize = parseFloat(priceFilter.tickSize);
      }

      this.exchangeInfoCache[symbol] = { lotStepSize, tickSize };
      return { lotStepSize, tickSize };
    } catch (error) {
      console.error('[BinanceClient] Failed to get exchange info:', error);
      return { lotStepSize: 0.00001, tickSize: 0.01 };
    }
  }

  private roundToStep(value: number, step: number): number {
    if (step <= 0) return value;
    const precision = Math.max(0, Math.ceil(-Math.log10(step)));
    return parseFloat((Math.floor(value / step) * step).toFixed(precision));
  }

  async roundQuantity(symbol: string, quantity: number): Promise<number> {
    const info = await this.getExchangeInfo(symbol);
    return this.roundToStep(quantity, info.lotStepSize);
  }

  async roundPrice(symbol: string, price: number): Promise<number> {
    const info = await this.getExchangeInfo(symbol);
    return this.roundToStep(price, info.tickSize);
  }

  // ========================================================================
  // WEBSOCKET SUBSCRIPTIONS (with reconnection)
  // ========================================================================

  private createReconnectingWs(url: string, onMessage: (event: MessageEvent) => void): WebSocket {
    let reconnectDelay = 3000;
    const maxDelay = 30000;
    let shouldReconnect = true;

    const connect = (): WebSocket => {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        console.log(`[BinanceClient WS] Connected: ${url.split('/').pop()}`);
        reconnectDelay = 3000; // reset on successful connect
      };

      ws.onmessage = onMessage;

      ws.onerror = (event: Event) => {
        const errAny = event as any;
        if (errAny.message && errAny.message.length > 0) {
          console.error(`[BinanceClient WS] Error:`, errAny.message);
        }
      };

      ws.onclose = () => {
        console.log(`[BinanceClient WS] Disconnected: ${url.split('/').pop()}`);
        if (shouldReconnect) {
          console.log(`[BinanceClient WS] Reconnecting in ${reconnectDelay / 1000}s...`);
          setTimeout(() => {
            if (shouldReconnect) connect();
          }, reconnectDelay);
          reconnectDelay = Math.min(reconnectDelay * 1.5, maxDelay);
        }
      };

      return ws;
    };

    const ws = connect();
    // Patch close to disable reconnect
    const originalClose = ws.close.bind(ws);
    ws.close = () => {
      shouldReconnect = false;
      originalClose();
    };
    return ws;
  }

  subscribeToTicker(symbol: string, callback: (data: TickerData) => void) {
    const url = `${this.websocketUrl}/${symbol.toLowerCase()}@ticker`;
    return this.createReconnectingWs(url, (event) => {
      const data = JSON.parse(event.data as string);
      callback({
        symbol: data.s,
        price: parseFloat(data.c),
        time: data.E,
      });
    });
  }

  subscribeToKline(
    symbol: string,
    interval: string,
    callback: (data: KlineData) => void
  ) {
    const url = `${this.websocketUrl}/${symbol.toLowerCase()}@kline_${interval}`;
    return this.createReconnectingWs(url, (event) => {
      const data = JSON.parse(event.data as string);
      const k = data.k;
      callback({
        time: k.t,
        open: parseFloat(k.o),
        high: parseFloat(k.h),
        low: parseFloat(k.l),
        close: parseFloat(k.c),
        volume: parseFloat(k.v),
      });
    });
  }

  subscribeToMultiStream(symbols: string[], streams: string[], callback: (data: any) => void) {
    const streamString = symbols
      .map((s) => streams.map((stream) => `${s.toLowerCase()}@${stream}`))
      .flat()
      .join('/');

    const url = `${this.websocketUrl}/stream?streams=${streamString}`;
    return this.createReconnectingWs(url, (event) => {
      const data = JSON.parse(event.data as string);
      callback(data.data);
    });
  }
}

export const createBinanceClient = (testnet: boolean = false): BinanceClient => {
  const apiKey = process.env.BINANCE_API_KEY;
  const apiSecret = process.env.BINANCE_API_SECRET;

  if (!apiKey || !apiSecret) {
    throw new Error('Binance API credentials not configured');
  }

  return new BinanceClient({
    apiKey,
    apiSecret,
    testnet,
  });
};
