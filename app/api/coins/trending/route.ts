import { coinGeckoService } from '@/lib/coingecko-service'

export async function GET() {
  try {
    const trendingCoins = await coinGeckoService.getTrendingCoins()

    return Response.json(
      {
        success: true,
        data: trendingCoins,
      },
      {
        status: 200,
        headers: {
          'Cache-Control': 'public, s-maxage=600, stale-while-revalidate=1200',
        },
      }
    )
  } catch (error) {
    console.error('Error fetching trending coins:', error)
    return Response.json(
      {
        success: false,
        error: 'Failed to fetch trending coins',
      },
      { status: 500 }
    )
  }
}
