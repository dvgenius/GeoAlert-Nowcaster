import React, { useState, useEffect } from 'react';
import { Radio, Activity, MapPin, Cpu, ShieldAlert, Globe } from 'lucide-react';

const REGION_OPTIONS = [
  { id: 'uttarakhand', name: 'Uttarakhand (Chamoli Basin)' },
  { id: 'mumbai', name: 'Mumbai (MMR / Coastal)' },
  { id: 'western_ghats', name: 'Western Ghats (Wayanad)' },
  { id: 'northeast', name: 'North-East (Sohra / Khasi)' },
];

export default function Navbar({ backendConnected, region, onChangeRegion, regionInfo }) {
  const [timeStr, setTimeStr] = useState('');

  useEffect(() => {
    const update = () => {
      const now = new Date();
      setTimeStr(now.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata' }) + ' IST');
    };
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, []);

  const centerLat = regionInfo?.center?.lat ?? 30.45;
  const centerLon = regionInfo?.center?.lon ?? 79.55;

  return (
    <header className="glass-panel border-b border-slate-800/80 px-6 py-3.5 sticky top-0 z-50">
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 shadow-lg shadow-blue-500/25">
            <Radio className="w-5 h-5 text-white animate-pulse" />
            <span className="absolute -top-1 -right-1 flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
            </span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-white font-heading">
                GeoAlert <span className="text-cyan-400">Nowcaster</span>
              </h1>
            </div>
            <p className="text-xs text-slate-400">
              AI-Driven Hyper-Local Early Warning System • Lead Time: 2–6 Hours
            </p>
          </div>
        </div>

        {/* Telemetry Chips & Region Selector */}
        <div className="flex flex-wrap items-center gap-3 text-xs">
          {/* Pan-India Region Selector */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-cyan-500/40 text-slate-200">
            <Globe className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-semibold text-cyan-300">Target Region:</span>
            <select
              value={region}
              onChange={(e) => onChangeRegion(e.target.value)}
              className="bg-slate-900 text-white font-medium text-xs px-2 py-0.5 rounded border border-slate-700 outline-none cursor-pointer hover:border-cyan-400 focus:border-cyan-400"
            >
              {REGION_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.name}
                </option>
              ))}
            </select>
            <span className="font-mono text-[11px] text-slate-400 hidden sm:inline">
              ({centerLat.toFixed(2)}°N, {centerLon.toFixed(2)}°E)
            </span>
          </div>

          {/* Model Status */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/60 border border-slate-700/60 text-slate-300">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span>ConvLSTM U-Net</span>
            <span className="px-1.5 py-0.2 bg-emerald-500/20 text-emerald-400 rounded text-[10px] font-bold">Pan-India PINN</span>
          </div>

          {/* Backend Status */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/60 border border-slate-700/60">
            <span className={`w-2 h-2 rounded-full ${backendConnected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'}`}></span>
            <span className={backendConnected ? 'text-emerald-300 font-medium' : 'text-rose-300 font-medium'}>
              {backendConnected ? 'FastAPI Ingestion Live' : 'Backend Offline'}
            </span>
          </div>

          {/* Clock */}
          <div className="font-mono px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-700/40 text-cyan-300">
            {timeStr}
          </div>
        </div>
      </div>
    </header>
  );
}
