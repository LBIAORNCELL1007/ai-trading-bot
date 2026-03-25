import BinanceTradingDashboard from "@/components/binance-trading-dashboard"

export default async function TradeRoute({ params }: { params: Promise<{ symbol: string }> }) {
  // Extract and await the params object (Next.js 15 standard)
  const resolvedParams = await params;
  const decodedSymbol = decodeURIComponent(resolvedParams.symbol)

  return (
    <main className="min-h-screen bg-[#121212]">
      <BinanceTradingDashboard initialSymbol={decodedSymbol} />
    </main>
  )
}
