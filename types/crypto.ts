export interface CoinData {
  id: string
  symbol: string
  name: string
  image: string
  currentPrice: number
  marketCap: number | null
  marketCapRank: number | null
  volume24h: number | null
  high24h: number | null
  low24h: number | null
  priceChange24h: number | null
  priceChangePercentage24h: number | null
  priceChangePercentage7d: number | null
  priceChangePercentage30d: number | null
  priceChangePercentage1y: number | null
  marketCapChange24h: number | null
  marketCapChangePercentage24h: number | null
  circulatingSupply: number | null
  totalSupply: number | null
  maxSupply: number | null
  ath: number | null
  athChangePercentage: number | null
  atl: number | null
  atlChangePercentage: number | null
  description?: string
  genesisDate?: string
  homepage?: string
  category?: string
  platform?: Record<string, string>
}

export interface MarketData {
  price: number
  marketCap: number
  volume24h: number
  timestamp: number
}

export interface PriceHistory {
  timestamp: number
  price: number
}

export interface ChartData {
  timestamp: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface TimeRange {
  label: string
  value: string | number
  days?: number
}

export interface CoinDetailResponse {
  id: string
  symbol: string
  name: string
  description?: {
    en?: string
  }
  links?: {
    homepage?: string[]
    blockchain_site?: string[]
  }
  image?: {
    large?: string
    small?: string
    thumb?: string
  }
  genesis_date?: string
  market_data?: {
    current_price?: { usd: number }
    market_cap?: { usd: number }
    market_cap_rank?: number
    total_volume?: { usd: number }
    high_24h?: { usd: number }
    low_24h?: { usd: number }
    price_change_24h?: number
    price_change_percentage_24h?: number
    circulating_supply?: number
    total_supply?: number
    max_supply?: number
    ath?: { usd: number }
    atl?: { usd: number }
  }
  community_data?: {
    facebook_likes?: number
    twitter_followers?: number
    reddit_average_posts_48h?: number
    reddit_average_comments_48h?: number
    reddit_subscribers?: number
  }
}

export interface GlobalData {
  data?: {
    active_cryptocurrencies?: number
    markets?: number
    total_market_cap?: { usd: number }
    total_volume_24h?: { usd: number }
    bitcoin_dominance?: number
    ethereum_dominance?: number
    market_cap_percentage?: Record<string, number>
  }
}
