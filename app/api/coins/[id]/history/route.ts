import { coinGeckoService } from '@/lib/coingecko-service'
import type { NextRequest } from 'next/server'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params
    const days = request.nextUrl.searchParams.get('days') || 'max'

    const priceHistory = await coinGeckoService.getPriceHistory(id, days)

    return Response.json(
      {
        success: true,
        data: priceHistory,
        timeRange: days,
      },
      {
        status: 200,
        headers: {
          'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=600',
        },
      }
    )
  } catch (error) {
    console.error(`Error fetching price history for ${params}:`, error)
    return Response.json(
      {
        success: false,
        error: 'Failed to fetch price history',
      },
      { status: 500 }
    )
  }
}
