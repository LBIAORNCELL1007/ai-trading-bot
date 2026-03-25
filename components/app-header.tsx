'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { BarChart3, Settings } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function AppHeader() {
  const pathname = usePathname()

  const isActive = (href: string) => pathname === href || pathname?.startsWith(href)

  return (
    /* THEME UPDATE: 
       - bg-[#121212]/70: Matte Black at 70% opacity 
       - backdrop-blur: Blur effect for the "Glass" look
       - border-[#1DB954]/30: Subtle Energetic Green border
    */
    <header className="sticky top-0 z-50 w-full border-b border-[#1DB954]/30 bg-[#121212]/70 backdrop-blur-md">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">

        {/* Logo / Brand: Updated to your Energetic Green */}
        <Link href="/" className="flex items-center gap-2 text-xl font-bold text-[#1DB954] hover:opacity-80 transition-opacity">
          <BarChart3 className="w-6 h-6" />
          <span className="hidden sm:inline tracking-tight">CryptoTracker</span>
        </Link>

        {/* Navigation */}
        <nav className="flex items-center gap-2">
          <Button
            variant="ghost"
            asChild
            /* Conditional Styling: 
               If active, it glows Green. If not, it's Soft Gray.
            */
            className={`gap-2 transition-all ${isActive('/crypto')
                ? 'bg-[#1DB954]/20 text-[#1DB954] border border-[#1DB954]/50'
                : 'text-[#EAEAEA]/70 hover:text-[#1DB954] hover:bg-[#1DB954]/10'
              }`}
          >
            <Link href="/crypto">
              <BarChart3 className="w-4 h-4" />
              <span className="hidden sm:inline">Coins</span>
            </Link>
          </Button>

          <Button
            variant="ghost"
            asChild
            className={`gap-2 transition-all ${isActive('/settings')
                ? 'bg-[#1DB954]/20 text-[#1DB954] border border-[#1DB954]/50'
                : 'text-[#EAEAEA]/70 hover:text-[#1DB954] hover:bg-[#1DB954]/10'
              }`}
          >
            <Link href="/settings">
              <Settings className="w-4 h-4" />
              <span className="hidden sm:inline">Settings</span>
            </Link>
          </Button>
        </nav>
      </div>
    </header>
  )
}