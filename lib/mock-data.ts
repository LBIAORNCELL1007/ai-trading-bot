// Generate initial mock price data
export const mockPriceData = Array.from({ length: 20 }, (_, i) => {
  const basePrice = 30000
  const randomVariation = (Math.random() - 0.5) * 1000
  return {
    time: new Date(Date.now() - (20 - i) * 3 * 60000).toISOString(),
    price: basePrice + randomVariation,
  }
})

// Function to generate a new price based on the current price
export function generateNewPrice(currentPrice: number) {
  // Random variation between -2% and +2%
  const randomPercentage = (Math.random() - 0.5) * 4
  const newPrice = currentPrice * (1 + randomPercentage / 100)

  // Ensure price doesn't go below a minimum value
  const finalPrice = Math.max(newPrice, 10000)

  // Calculate the percentage change
  const percentageChange = Math.abs(((finalPrice - currentPrice) / currentPrice) * 100)

  // Determine direction
  const direction = finalPrice >= currentPrice ? "up" : "down"

  return {
    newPrice: finalPrice,
    newChange: percentageChange,
    direction: direction as "up" | "down",
  }
}
