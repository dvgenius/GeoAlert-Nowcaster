import React, { useState } from 'react';
import { Eye, HelpCircle, ChevronDown, ChevronUp, Sparkles, BookOpen } from 'lucide-react';

const FACTOR_COLORS = {
  IWV: { color: '#38bdf8', gradient: 'from-sky-500 to-blue-600', badge: 'Moisture' },
  CAPE: { color: '#f59e0b', gradient: 'from-amber-400 to-orange-500', badge: 'Instability' },
  CIN: { color: '#a855f7', gradient: 'from-purple-500 to-indigo-600', badge: 'Cap Trigger' },
  CTT_DROP: { color: '#ef4444', gradient: 'from-rose-500 to-red-600', badge: 'Updraft' },
  DEM: { color: '#10b981', gradient: 'from-emerald-400 to-teal-600', badge: 'Topography' },
};

export default function XaiDrawer({ xai }) {
  const [isOpen, setIsOpen] = useState(true);

  if (!xai || !xai.breakdown) return null;

  const breakdownList = Object.entries(xai.breakdown).sort(
    (a, b) => b[1].percentage - a[1].percentage
  );

  return (
    <div className="glass-panel rounded-2xl border border-slate-700/60 p-4 shadow-xl">
      {/* Header with Collapsible Toggle */}
      <div
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between cursor-pointer select-none"
      >
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-cyan-500/20 text-cyan-400">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white font-heading">
              Explainable AI (XAI) Attribution
            </h3>
            <p className="text-[11px] text-slate-400">
              Captum Integrated Gradients • Feature Breakdown
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
            PINN Physics Checked
          </span>
          <button className="text-slate-400 hover:text-white p-1">
            {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {isOpen && (
        <div className="mt-3.5 space-y-3 pt-3 border-t border-slate-800/80">
          {/* Narrative Box */}
          {xai.narrative && (
            <div className="p-3 rounded-xl bg-slate-800/50 border border-slate-700/50 text-xs text-slate-300 leading-relaxed">
              <span className="font-semibold text-cyan-300 flex items-center gap-1 mb-1">
                <BookOpen className="w-3.5 h-3.5" /> Thermodynamic Narrative:
              </span>
              {xai.narrative}
            </div>
          )}

          {/* Feature Progress Bars */}
          <div className="space-y-2.5">
            {breakdownList.map(([key, item]) => {
              const meta = FACTOR_COLORS[key] || {
                color: '#38bdf8',
                gradient: 'from-blue-500 to-cyan-500',
                badge: 'Atmosphere'
              };

              return (
                <div key={key} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-1.5">
                      <span
                        className="w-2 h-2 rounded-full"
                        style={{ backgroundColor: meta.color }}
                      ></span>
                      <span className="font-semibold text-slate-200">{key}</span>
                      <span className="text-[10px] text-slate-400 hidden sm:inline">
                        — {item.full_name}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-slate-400">
                        {meta.badge}
                      </span>
                      <span className="font-mono font-bold text-white text-xs">
                        {item.percentage.toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Progress Bar */}
                  <div className="w-full bg-slate-800/90 rounded-full h-2 overflow-hidden">
                    <div
                      className={`h-full rounded-full bg-gradient-to-r ${meta.gradient} transition-all duration-700`}
                      style={{ width: `${Math.max(4, item.percentage)}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="text-[10px] text-slate-500 text-right pt-1 font-mono">
            Attributions computed across 4 time steps and 64x64 grid
          </div>
        </div>
      )}
    </div>
  );
}
