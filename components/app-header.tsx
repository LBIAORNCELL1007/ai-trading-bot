'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { BarChart3, Settings, Home } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function AppHeader() {
  const pathname = usePathname()

  const isActive = (href: string) => pathname === href || pathname?.startsWith(href)

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        {/* Logo / Brand */}
        <Link href="/" className="flex items-center gap-2 text-xl font-bold text-primary hover:opacity-80 transition-opacity">
          <BarChart3 className="w-6 h-6" />
          <span className="hidden sm:inline">CryptoTracker</span>
        </Link>

        {/* Navigation */}
        <nav className="flex items-center gap-2">
          <Button
            variant={isActive('/crypto') ? 'default' : 'ghost'}
            asChild
            className="gap-2"
          >
            <Link href="/crypto">
              <BarChart3 className="w-4 h-4" />
              <span className="hidden sm:inline">Coins</span>
            </Link>
          </Button>

          <Button
            variant={isActive('/settings') ? 'default' : 'ghost'}
            asChild
            className="gap-2"
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
