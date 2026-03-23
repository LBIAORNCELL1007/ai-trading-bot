import { coinGeckoService } from '@/lib/coingecko-service'
import type { NextRequest } from 'next/server'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params
    const coinDetails = await coinGeckoService.getCoinDetails(id)

    return Response.json(
      {
        success: true,
        data: coinDetails,
      },
      {
        status: 200,
        headers: {
          'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=600',
        },
      }
    )
  } catch (error) {
    console.error(`Error fetching coin details for ${params}:`, error)
    return Response.json(
      {
        success: false,
        error: 'Failed to fetch coin details',
      },
      { status: 500 }
    )
  }
}
