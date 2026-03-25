import type React from "react"
import "./globals.css"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { AppHeader } from "@/components/app-header"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "CryptoTracker - AI Trading Bot",
  description: "Modern AI Trading Interface",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.className} antialiased`}
        style={{
          backgroundColor: '#121212', // Your Matte Black/Charcoal
          color: '#EAEAEA',
          margin: 0,
          minHeight: '100vh'
        }}
      >
        <AppHeader />
        <main>{children}</main>
      </body>
    </html>
  )
}