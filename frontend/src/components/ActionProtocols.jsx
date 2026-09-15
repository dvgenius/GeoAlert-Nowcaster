import React from 'react';
import { PhoneCall, CheckCircle2, Navigation, AlertCircle } from 'lucide-react';

export default function ActionProtocols({ alertCategory }) {
  const isRed = alertCategory === 'RED';

  return (
    <div className="glass-panel rounded-2xl p-4 border border-slate-700/60 shadow-lg space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-white font-heading flex items-center gap-2">
          <Navigation className="w-4 h-4 text-cyan-400" />
          Emergency Protocols & Staging
        </h3>
        <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono">
          SDRF / DDMA Chamoli
        </span>
      </div>

      <div className="space-y-2 text-xs">
        <div className="p-2.5 rounded-xl bg-slate-800/60 border border-slate-700/40 flex items-start gap-2.5">
          <CheckCircle2 className={`w-4 h-4 mt-0.5 shrink-0 ${isRed ? 'text-red-400' : 'text-emerald-400'}`} />
          <div>
            <span className="font-semibold text-slate-200">Valley Inundation Evacuation:</span>
            <p className="text-slate-400 text-[11px] mt-0.5">
              Clear low-lying settlements along Alaknanda, Dhauliganga & Rishi Ganga river corridors. Relocate to designated high-ground shelters in Joshimath and Pipalkoti.
            </p>
          </div>
        </div>

        <div className="p-2.5 rounded-xl bg-slate-800/60 border border-slate-700/40 flex items-start gap-2.5">
          <CheckCircle2 className={`w-4 h-4 mt-0.5 shrink-0 ${isRed ? 'text-red-400' : 'text-amber-400'}`} />
          <div>
            <span className="font-semibold text-slate-200">Pilgrim & Tourist Traffic Control:</span>
            <p className="text-slate-400 text-[11px] mt-0.5">
              Halt pilgrim convoy movement along Badrinath National Highway (NH-07). Restrict vehicular entry near landslide-prone gorge bottleneck stretches.
            </p>
          </div>
        </div>

        <div className="p-2.5 rounded-xl bg-slate-800/60 border border-slate-700/40 flex items-start gap-2.5">
          <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0 text-cyan-400" />
          <div>
            <span className="font-semibold text-slate-200">NDRF / SDRF Pre-Positioning:</span>
            <p className="text-slate-400 text-[11px] mt-0.5">
              Standby swift-water rescue rafts and heavy earth-moving equipment at Chamoli District HQ and Gauchar airstrip.
            </p>
          </div>
        </div>
      </div>

      {/* Emergency Contact Bar */}
      <div className="pt-2 border-t border-slate-800 flex items-center justify-between text-[11px]">
        <span className="text-slate-400 flex items-center gap-1">
          <PhoneCall className="w-3 h-3 text-emerald-400" /> District Emergency:
        </span>
        <span className="font-mono font-bold text-cyan-300">1077 / 01372-251077</span>
      </div>
    </div>
  );
}
