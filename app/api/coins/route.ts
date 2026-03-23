import { coinGeckoService } from '@/lib/coingecko-service'
import type { NextRequest } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const page = parseInt(searchParams.get('page') || '1', 10)
    const perPage = Math.min(parseInt(searchParams.get('perPage') || '250', 10), 250)

    const coins = await coinGeckoService.getAllCoins(page, perPage)

    return Response.json(
      {
        success: true,
        data: coins,
        pagination: {
          page,
          perPage,
          total: coins.length,
        },
      },
      {
        status: 200,
        headers: {
          'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=600', // Cache for 5 minutes
        },
      }
    )
  } catch (error) {
    console.error('Error fetching coins:', error)
    
    // Return cached error response on rate limit
    if (error instanceof Error && error.message.includes('429')) {
      return Response.json(
        {
          success: false,
          error: 'API rate limit reached. Using cached data.',
        },
        { 
          status: 200,
          headers: {
            'Cache-Control': 'public, s-maxage=60, stale-while-revalidate=300',
          },
        }
      )
    }

    return Response.json(
      {
        success: false,
        error: 'Failed to fetch coins',
      },
      { status: 500 }
    )
  }
}
