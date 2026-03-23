"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface TradingInterfaceProps {
  currentPrice: number
  balance: {
    usd: number
    btc: number
  }
  setBalance: React.Dispatch<
    React.SetStateAction<{
      usd: number
      btc: number
    }>
  >
  setTrades: React.Dispatch<React.SetStateAction<any[]>>
}

export default function TradingInterface({ currentPrice, balance, setBalance, setTrades }: TradingInterfaceProps) {
  const [buyAmount, setBuyAmount] = useState("")
  const [sellAmount, setSellAmount] = useState("")

  const handleBuy = () => {
    const amount = Number.parseFloat(buyAmount)
    if (isNaN(amount) || amount <= 0) return

    const cost = amount * currentPrice
    if (cost > balance.usd) return

    setBalance((prev) => ({
      usd: prev.usd - cost,
      btc: prev.btc + amount,
    }))

    setTrades((prev) => [
      ...prev,
      {
        id: Date.now(),
        type: "buy",
        amount,
        price: currentPrice,
        total: cost,
        time: new Date().toISOString(),
      },
    ])

    setBuyAmount("")
  }

  const handleSell = () => {
    const amount = Number.parseFloat(sellAmount)
    if (isNaN(amount) || amount <= 0 || amount > balance.btc) return

    const value = amount * currentPrice

    setBalance((prev) => ({
      usd: prev.usd + value,
      btc: prev.btc - amount,
    }))

    setTrades((prev) => [
      ...prev,
      {
        id: Date.now(),
        type: "sell",
        amount,
        price: currentPrice,
        total: value,
        time: new Date().toISOString(),
      },
    ])

    setSellAmount("")
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Manual Trading</CardTitle>
        <CardDescription>Buy or sell BTC</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="buy" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="buy">Buy</TabsTrigger>
            <TabsTrigger value="sell">Sell</TabsTrigger>
          </TabsList>

          <TabsContent value="buy" className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="buy-amount">Amount (BTC)</Label>
              <Input
                id="buy-amount"
                type="number"
                placeholder="0.00"
                value={buyAmount}
                onChange={(e) => setBuyAmount(e.target.value)}
              />
            </div>

            <div className="text-sm">
              <div className="flex justify-between">
                <span>Price:</span>
                <span>${currentPrice.toLocaleString()}</span>
              </div>
              <div className="flex justify-between mt-1">
                <span>Total Cost:</span>
                <span>${(Number.parseFloat(buyAmount) || 0) * currentPrice}</span>
              </div>
              <div className="flex justify-between mt-1">
                <span>Available:</span>
                <span>${balance.usd.toLocaleString()}</span>
              </div>
            </div>

            <Button
              className="w-full"
              onClick={handleBuy}
              disabled={
                !buyAmount ||
                Number.parseFloat(buyAmount) <= 0 ||
                Number.parseFloat(buyAmount) * currentPrice > balance.usd
              }
            >
              Buy BTC
            </Button>
          </TabsContent>

          <TabsContent value="sell" className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="sell-amount">Amount (BTC)</Label>
              <Input
                id="sell-amount"
                type="number"
                placeholder="0.00"
                value={sellAmount}
                onChange={(e) => setSellAmount(e.target.value)}
              />
            </div>

            <div className="text-sm">
              <div className="flex justify-between">
                <span>Price:</span>
                <span>${currentPrice.toLocaleString()}</span>
              </div>
              <div className="flex justify-between mt-1">
                <span>Total Value:</span>
                <span>${(Number.parseFloat(sellAmount) || 0) * currentPrice}</span>
              </div>
              <div className="flex justify-between mt-1">
                <span>Available:</span>
                <span>{balance.btc.toFixed(8)} BTC</span>
              </div>
            </div>

            <Button
              className="w-full"
              onClick={handleSell}
              disabled={
                !sellAmount || Number.parseFloat(sellAmount) <= 0 || Number.parseFloat(sellAmount) > balance.btc
              }
            >
              Sell BTC
            </Button>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
