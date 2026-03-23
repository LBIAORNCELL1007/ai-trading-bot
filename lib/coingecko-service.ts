import type { CoinData, MarketData, PriceHistory } from '@/types/crypto'

const COINGECKO_API = 'https://api.coingecko.com/api/v3'
const CACHE_DURATION = 60 // 60 seconds

interface CoinGeckoMarketData {
  id: string
  symbol: string
  name: string
  image: string
  current_price: number | null
  market_cap: number | null
  market_cap_rank: number | null
  total_volume: number | null
  high_24h: number | null
  low_24h: number | null
  price_change_24h: number | null
  price_change_percentage_24h: number | null
  price_change_percentage_7d_in_currency: number | null
  price_change_percentage_30d_in_currency: number | null
  price_change_percentage_1y_in_currency: number | null
  market_cap_change_24h: number | null
  market_cap_change_percentage_24h: number | null
  circulating_supply: number | null
  total_supply: number | null
  max_supply: number | null
  ath: number | null
  ath_change_percentage: number | null
  atl: number | null
  atl_change_percentage: number | null
}

class CoinGeckoService {
  private cache: Map<string, { data: any; timestamp: number }> = new Map()

  private getCacheKey(endpoint: string, params: Record<string, any> = {}): string {
    const paramStr = Object.entries(params)
      .sort()
      .map(([k, v]) => `${k}=${v}`)
      .join('&')
    return `${endpoint}:${paramStr}`
  }

  private isCacheValid(cacheKey: string): boolean {
    const cached = this.cache.get(cacheKey)
    if (!cached) return false
    return Date.now() - cached.timestamp < CACHE_DURATION * 1000
  }

  private async fetchWithCache<T>(endpoint: string, params: Record<string, any> = {}): Promise<T> {
    const cacheKey = this.getCacheKey(endpoint, params)

    if (this.isCacheValid(cacheKey)) {
      return this.cache.get(cacheKey)!.data as T
    }

    try {
      const queryParams = new URLSearchParams(
        Object.entries(params).reduce(
          (acc, [k, v]) => {
            acc[k] = String(v)
            return acc
          },
          {} as Record<string, string>
        )
      )

      const response = await fetch(`${COINGECKO_API}${endpoint}?${queryParams}`, {
        next: { revalidate: CACHE_DURATION },
      })

      if (!response.ok) {
        // If we have older cache data, return it on rate limit
        const expired = this.cache.get(cacheKey)
        if (expired && response.status === 429) {
          console.warn(`Rate limited on ${endpoint}, returning stale cache`)
          return expired.data as T
        }
        throw new Error(`CoinGecko API error: ${response.status}`)
      }

      const data = await response.json()
      this.cache.set(cacheKey, { data, timestamp: Date.now() })

      return data as T
    } catch (error) {
      console.error(`CoinGecko API request failed for ${endpoint}:`, error)
      throw error
    }
  }

  async getAllCoins(page = 1, perPage = 250): Promise<CoinData[]> {
    const data = await this.fetchWithCache<CoinGeckoMarketData[]>('/coins/markets', {
      vs_currency: 'usd',
      order: 'market_cap_desc',
      per_page: perPage,
      page,
      sparkline: false,
      price_change_percentage: '1h,24h,7d,30d,1y',
      locale: 'en',
    })

    return data.map((coin) => this.transformCoinData(coin))
  }

  async searchCoins(query: string): Promise<Array<{ id: string; name: string; symbol: string; image: string }>> {
    const data = await this.fetchWithCache<any>('/search', { query })
    return data.coins.slice(0, 20).map((coin: any) => ({
      id: coin.id,
      name: coin.name,
      symbol: coin.symbol,
      image: coin.thumb,
    }))
  }

  async getCoinDetails(coinId: string): Promise<any> {
    return this.fetchWithCache(`/coins/${coinId}`, {
      localization: false,
      tickers: false,
      market_data: true,
      community_data: false,
      developer_data: false,
      sparkline: false,
    })
  }

  async getPriceHistory(coinId: string, days: number | string = 'max'): Promise<PriceHistory[]> {
    const data = await this.fetchWithCache<any>(`/coins/${coinId}/market_chart`, {
      vs_currency: 'usd',
      days,
    })

    return data.prices.map((point: [number, number]) => ({
      timestamp: point[0],
      price: point[1],
    }))
  }

  async getOHLCData(coinId: string, days: number): Promise<any[]> {
    return this.fetchWithCache(`/coins/${coinId}/ohlc`, {
      vs_currency: 'usd',
      days,
    })
  }

  async getTrendingCoins(): Promise<CoinData[]> {
    const data = await this.fetchWithCache<any>('/search/trending')
    const coinIds = data.coins.slice(0, 7).map((item: any) => item.item.id)
    return this.getAllCoins(1, 250).then((coins) => coins.filter((c) => coinIds.includes(c.id)))
  }

  async getGlobalData(): Promise<any> {
    return this.fetchWithCache('/global')
  }

  private transformCoinData(coin: CoinGeckoMarketData): CoinData {
    return {
      id: coin.id,
      symbol: coin.symbol.toUpperCase(),
      name: coin.name,
      image: coin.image,
      currentPrice: coin.current_price || 0,
      marketCap: coin.market_cap,
      marketCapRank: coin.market_cap_rank,
      volume24h: coin.total_volume,
      high24h: coin.high_24h,
      low24h: coin.low_24h,
      priceChange24h: coin.price_change_24h,
      priceChangePercentage24h: coin.price_change_percentage_24h,
      priceChangePercentage7d: coin.price_change_percentage_7d_in_currency,
      priceChangePercentage30d: coin.price_change_percentage_30d_in_currency,
      priceChangePercentage1y: coin.price_change_percentage_1y_in_currency,
      marketCapChange24h: coin.market_cap_change_24h,
      marketCapChangePercentage24h: coin.market_cap_change_percentage_24h,
      circulatingSupply: coin.circulating_supply,
      totalSupply: coin.total_supply,
      maxSupply: coin.max_supply,
      ath: coin.ath,
      athChangePercentage: coin.ath_change_percentage,
      atl: coin.atl,
      atlChangePercentage: coin.atl_change_percentage,
    }
  }
}

export const coinGeckoService = new CoinGeckoService()
