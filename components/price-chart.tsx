"use client"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

interface PriceChartProps {
  data: Array<{
    time: string
    price: number
  }>
}

export default function PriceChart({ data }: PriceChartProps) {
  const formattedData = data.map((item) => ({
    time: new Date(item.time).toLocaleTimeString(),
    price: item.price,
  }))

  return (
    <div className="w-full h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={formattedData}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => {
              return value.split(":").slice(0, 2).join(":")
            }}
          />
          <YAxis
            domain={["auto", "auto"]}
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => `$${value.toLocaleString()}`}
          />
          <Tooltip
            formatter={(value: any) => [`$${Number(value).toLocaleString()}`, "Price"]}
            labelFormatter={(label) => `Time: ${label}`}
          />
          <Line type="monotone" dataKey="price" stroke="#3b82f6" strokeWidth={2} dot={false} activeDot={{ r: 6 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
