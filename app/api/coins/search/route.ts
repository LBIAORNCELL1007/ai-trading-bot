import { coinGeckoService } from '@/lib/coingecko-service'
import type { NextRequest } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    const query = request.nextUrl.searchParams.get('q') || ''

    if (!query || query.length < 1) {
      return Response.json(
        {
          success: false,
          error: 'Query parameter is required',
        },
        { status: 400 }
      )
    }

    const results = await coinGeckoService.searchCoins(query)

    return Response.json(
      {
        success: true,
        data: results,
      },
      {
        status: 200,
        headers: {
          'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=600',
        },
      }
    )
  } catch (error) {
    console.error('Error searching coins:', error)
    return Response.json(
      {
        success: false,
        error: 'Failed to search coins',
      },
      { status: 500 }
    )
  }
}
