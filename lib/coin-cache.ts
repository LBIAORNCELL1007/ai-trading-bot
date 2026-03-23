// Client-side coin cache for optimized search without API rate limiting
let cachedCoins: Array<{
  id: string
  name: string
  symbol: string
  image: string
  current_price?: number
}> = []

let cacheLoadedPromise: Promise<void> | null = null
let isCacheLoading = false

export async function loadCoinCache() {
  if (cachedCoins.length > 0) return
  if (isCacheLoading) return cacheLoadedPromise

  isCacheLoading = true
  cacheLoadedPromise = (async () => {
    try {
      // Fetch coins in batches to avoid overwhelming the API
      const allCoins: typeof cachedCoins = []
      
      // Load top 3000 coins in 12 pages of 250 coins each
      for (let page = 1; page <= 12; page++) {
        try {
          const response = await fetch(`/api/coins?page=${page}&perPage=250`)
          if (!response.ok) break
          const data = await response.json()
          if (Array.isArray(data.data) && data.data.length > 0) {
            allCoins.push(
              ...data.data.map((coin: any) => ({
                id: coin.id,
                name: coin.name,
                symbol: coin.symbol,
                image: coin.image,
                current_price: coin.currentPrice,
              }))
            )
          }
        } catch (err) {
          console.error(`Error loading page ${page}:`, err)
          break
        }
      }

      cachedCoins = allCoins
      isCacheLoading = false
    } catch (error) {
      console.error('Failed to load coin cache:', error)
      isCacheLoading = false
    }
  })()

  await cacheLoadedPromise
}

export function searchCoinsLocally(
  query: string
): Array<{ id: string; name: string; symbol: string; image: string; current_price?: number }> {
  if (!query.trim() || cachedCoins.length === 0) return []

  const normalizedQuery = query.toLowerCase().trim()

  // Filter coins that match name or symbol
  return cachedCoins
    .filter((coin) => {
      const matchesName = coin.name.toLowerCase().includes(normalizedQuery)
      const matchesSymbol = coin.symbol.toLowerCase().includes(normalizedQuery)
      return matchesName || matchesSymbol
    })
    .sort((a, b) => {
      // Prioritize exact symbol matches
      if (a.symbol.toLowerCase() === normalizedQuery) return -1
      if (b.symbol.toLowerCase() === normalizedQuery) return 1

      // Then prioritize name starts with
      const aNameStarts = a.name.toLowerCase().startsWith(normalizedQuery)
      const bNameStarts = b.name.toLowerCase().startsWith(normalizedQuery)
      if (aNameStarts && !bNameStarts) return -1
      if (!aNameStarts && bNameStarts) return 1

      // Then prioritize symbol starts with
      const aSymbolStarts = a.symbol.toLowerCase().startsWith(normalizedQuery)
      const bSymbolStarts = b.symbol.toLowerCase().startsWith(normalizedQuery)
      if (aSymbolStarts && !bSymbolStarts) return -1
      if (!aSymbolStarts && bSymbolStarts) return 1

      return 0
    })
    .slice(0, 15)
}

export function getCachedCoins() {
  return cachedCoins
}

export function isCacheReady() {
  return cachedCoins.length > 0
}
