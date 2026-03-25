'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import {
    BarChart3, ShieldCheck, Zap, Brain, ArrowRight,
    Activity, Lock, Network, Users
} from 'lucide-react'

export default function LandingPage() {
    const [mounted, setMounted] = useState(false)

    useEffect(() => {
        setMounted(true)
    }, [])

    return (
        <div className="min-h-screen bg-[#121212] text-[#EAEAEA] selection:bg-[#1DB954]/30 font-sans relative overflow-hidden">

            {/* HERO SECTION WITH UPWARD TRENDING GRAPH */}
            <section className="relative pt-24 pb-32 z-10 border-b border-white/5">
                {/* Constant Upward Graph Background */}
                <div className="absolute inset-0 z-0 pointer-events-none opacity-25">
                    <UpwardTrendingGraph />
                </div>

                {/* Ambient Glow */}
                <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-[#1DB954]/5 blur-[120px] rounded-full" />

                <div className="container mx-auto px-6 text-center relative z-10">
                    <div className="inline-flex items-center gap-2 bg-[#1A1A1A] border border-[#1DB954]/30 text-[#1DB954] px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-widest mb-8">
                        <Activity className="w-3 h-3 animate-pulse" />
                        System Status: V3.0 Autonomous
                    </div>

                    <h1 className="text-5xl md:text-8xl font-black tracking-tighter mb-8 leading-tight">
                        QUANTITATIVE <br />
                        <span className="text-[#1DB954]">INTELLIGENCE</span>
                    </h1>

                    <p className="max-w-3xl mx-auto text-lg md:text-xl text-gray-400 mb-12 leading-relaxed">
                        A high-fidelity trading framework fusing <strong>Temporal Deep Learning (TCN)</strong>
                        with <strong>Topological Data Analysis (TDA)</strong> to navigate non-stationary
                        crypto regimes with institutional-grade risk parity.
                    </p>

                    <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
                        <Link href="/dashboard">
                            <Button size="lg" className="bg-[#1DB954] hover:bg-[#1AA34A] text-white font-black px-10 py-7 rounded-2xl gap-3 text-xl shadow-[0_0_40px_rgba(29,185,84,0.2)] transition-all hover:scale-105">
                                Launch Engine <ArrowRight className="w-6 h-6" />
                            </Button>
                        </Link>
                    </div>
                </div>
            </section>

            {/* Innovation Grid */}
            <section className="container mx-auto px-6 py-24 relative z-10">
                <div className="text-left mb-16">
                    <h2 className="text-3xl font-bold tracking-tight mb-2">Key Innovations</h2>
                    <p className="text-gray-500">Differentiating retail bots from institutional quantitative systems.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                    <FeatureCard
                        icon={<Brain className="w-8 h-8 text-[#1DB954]" />}
                        title="TDA Regime Detection"
                        description="Extracts persistent homology features to identify market loops and trend exhaustion voids."
                    />
                    <FeatureCard
                        icon={<Network className="w-8 h-8 text-[#1DB954]" />}
                        title="Global TCN Models"
                        description="Temporal Convolutional Networks trained on the entire crypto universe for cross-asset generalization."
                    />
                    <FeatureCard
                        icon={<Zap className="w-8 h-8 text-[#1DB954]" />}
                        title="Triple Barrier Labeling"
                        description="ML meta-labeling modeling TP, SL, and Time-Expiry barriers to reduce intraday noise."
                    />
                    <FeatureCard
                        icon={<Lock className="w-8 h-8 text-[#1DB954]" />}
                        title="HRP Risk Management"
                        description="Hierarchical Risk Parity and VaR at 95% confidence for scientific capital allocation."
                    />
                    <FeatureCard
                        icon={<Activity className="w-8 h-8 text-[#1DB954]" />}
                        title="Fractional Differencing"
                        description="Stationarity transformation (d=0.4) that maintains long-memory price dependencies."
                    />
                    <FeatureCard
                        icon={<ShieldCheck className="w-8 h-8 text-[#1DB954]" />}
                        title="Agentic Orchestration"
                        description="An autonomous decision-making layer that scans, validates, and manages position lifecycles."
                    />
                </div>
            </section>

            {/* RESEARCH CORE - CLEAN SINGLE ROW */}
            <section className="container mx-auto px-6 py-20 border-t border-white/5 relative z-10">
                <div className="flex items-center gap-2 text-[#1DB954] mb-12">
                    <Users className="w-5 h-5" />
                    <span className="font-bold uppercase tracking-widest text-sm">Research Core</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 border-l border-[#1DB954]/20 pl-6">
                    <TeamMember name="Sagnik Bhowmick" role="Bsc Data Science | 23BSD7045" univ="VIT-AP University" />
                    <TeamMember name="Mantri Krishna Sri Inesh" role="Bsc Data Science | 23BSD7023" univ="VIT-AP University" />
                    <TeamMember name="Sunkavalli LSVP SeshaSai" role="Bsc Data Science | 23BSD7019" univ="VIT-AP University" />
                </div>
            </section>

            {/* Engine Hardening Metrics */}
            <section className="container mx-auto px-6 py-24 bg-[#1A1A1A]/30 rounded-3xl border border-white/5 mb-24 relative z-10">
                <div className="flex flex-col md:flex-row gap-12 items-center">
                    <div className="flex-1">
                        <h2 className="text-4xl font-bold mb-6">Engine Hardening</h2>
                        <div className="space-y-4">
                            <StatusItem label="Critical Logic Errors Fixed" progress="100%" color="text-green-400" />
                            <StatusItem label="Institutional Workflow Integrated" progress="100%" color="text-green-400" />
                            <StatusItem label="TCN Global Weights Calibrated" progress="100%" color="text-green-400" />
                        </div>
                    </div>
                    <div className="flex-1 grid grid-cols-2 gap-4">
                        <StatBox label="Bug Fixes" value="19/24" sub="System Stability" />
                        <StatBox label="Asset Universe" value="TOP 30" sub="Liquid Pairs" />
                        <StatBox label="Execution" value="Paper" sub="Live Simulation" />
                        <StatBox label="Memory" value="FracDiff" sub="Stationary" />
                    </div>
                </div>
            </section>
        </div>
    )
}

function UpwardTrendingGraph() {
    return (
        <svg className="w-full h-full" viewBox="0 0 1440 400" preserveAspectRatio="none">
            <defs>
                <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#1DB954" stopOpacity="0" />
                    <stop offset="50%" stopColor="#1DB954" stopOpacity="0.8" />
                    <stop offset="100%" stopColor="#1DB954" stopOpacity="0" />
                </linearGradient>
            </defs>
            <path
                d="M0,380 L120,360 L240,370 L360,310 L480,330 L600,250 L720,270 L840,180 L960,200 L1080,120 L1200,140 L1440,40"
                fill="none"
                stroke="url(#lineGradient)"
                strokeWidth="3"
                strokeLinecap="round"
            >
                <animate
                    attributeName="stroke-dasharray"
                    from="0, 2000"
                    to="2000, 0"
                    dur="5s"
                    repeatCount="indefinite"
                />
            </path>
        </svg>
    )
}

function FeatureCard({ icon, title, description }: { icon: any, title: string, description: string }) {
    return (
        <div className="p-8 rounded-2xl bg-[#1A1A1A] border border-white/5 hover:border-[#1DB954]/40 transition-all group">
            <div className="mb-6 p-3 bg-[#121212] w-fit rounded-xl border border-white/5 group-hover:scale-110 group-hover:bg-[#1DB954]/10 transition-all">
                {icon}
            </div>
            <h3 className="text-xl font-bold mb-3">{title}</h3>
            <p className="text-gray-400 text-sm leading-relaxed">{description}</p>
        </div>
    )
}

function TeamMember({ name, role, univ }: { name: string, role: string, univ: string }) {
    return (
        <div>
            <h4 className="text-lg font-bold text-white mb-1">{name}</h4>
            <p className="text-xs text-[#1DB954] font-mono font-bold">{role}</p>
            <p className="text-[10px] text-gray-500 uppercase tracking-widest mt-1">{univ}</p>
        </div>
    )
}

function StatusItem({ label, progress, color }: { label: string, progress: string, color: string }) {
    return (
        <div className="flex justify-between items-center py-2 border-b border-white/5">
            <span className="text-sm font-medium">{label}</span>
            <span className={`text-xs font-bold font-mono ${color}`}>{progress}</span>
        </div>
    )
}

function StatBox({ label, value, sub }: { label: string, value: string, sub: string }) {
    return (
        <div className="bg-[#121212] p-6 rounded-2xl border border-white/5">
            <div className="text-gray-500 text-[10px] uppercase font-bold tracking-widest mb-1">{label}</div>
            <div className="text-2xl font-black text-[#1DB954]">{value}</div>
            <div className="text-gray-600 text-[10px] mt-1 italic">{sub}</div>
        </div>
    )
}