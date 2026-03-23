import type React from "react"
import "./globals.css"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { AppHeader } from "@/components/app-header"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "CryptoTracker - Live Cryptocurrency Prices & Market Data",
  description: "Track real-time cryptocurrency prices, market cap, volume, and supply data for 10,000+ coins including Bitcoin, Ethereum, and memecoins",
  keywords: "crypto, bitcoin, ethereum, cryptocurrency prices, market cap, trading, blockchain",
    generator: 'v0.app'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AppHeader />
        {children}
      </body>
    </html>
  )
}
