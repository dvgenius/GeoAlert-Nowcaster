import React, { useState, useEffect } from 'react';
import { Play, Pause, FastForward, Sliders, Layers } from 'lucide-react';

const PRESETS = [
  { hours: 2.0, label: '2h Initiation', tag: 'Cumulus Updraft' },
  { hours: 3.0, label: '3h Peak Convection', tag: 'Core Freezing' },
  { hours: 4.0, label: '4h Downburst', tag: 'Torrential Deluge' },
  { hours: 6.0, label: '6h Flood Runoff', tag: 'Valley Accumulation' },
];

export default function LeadTimeControl({ leadTime, onChangeLeadTime, isLoading }) {
  const [isPlaying, setIsPlaying] = useState(false);

  // Auto-play timeline animation
  useEffect(() => {
    let timer;
    if (isPlaying) {
      timer = setInterval(() => {
        onChangeLeadTime((prev) => {
          if (prev >= 6.0) return 2.0;
          const next = Math.round((prev + 0.5) * 10) / 10;
          return next > 6.0 ? 2.0 : next;
        });
      }, 3500);
    }
    return () => clearInterval(timer);
  }, [isPlaying, onChangeLeadTime]);

  return (
    <div className="glass-panel rounded-2xl p-4 border border-slate-700/60 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <span className="text-sm font-bold text-white font-heading">
            Nowcast Lead-Time Control
          </span>
          {isLoading && (
            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-cyan-500/20 text-cyan-300 animate-pulse">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping"></span>
              Computing AI Tensor...
            </span>
          )}
        </div>

        {/* Play/Pause Simulator */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              isPlaying
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 hover:bg-amber-500/30'
                : 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 hover:bg-indigo-500/30'
            }`}
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5 fill-current" />}
            <span>{isPlaying ? 'Pause Simulation' : 'Auto-Play Timeline'}</span>
          </button>
        </div>
      </div>

      {/* Range Slider */}
      <div className="px-2 mb-4">
        <div className="flex justify-between text-xs text-slate-400 font-mono mb-1.5">
          <span>T+2.0 Hours</span>
          <span className="text-cyan-400 font-bold text-sm">{leadTime.toFixed(1)} Hours Lead Time</span>
          <span>T+6.0 Hours</span>
        </div>
        <input
          type="range"
          min="2.0"
          max="6.0"
          step="0.5"
          value={leadTime}
          onChange={(e) => {
            setIsPlaying(false);
            onChangeLeadTime(parseFloat(e.target.value));
          }}
          className="w-full cursor-pointer"
        />
      </div>

      {/* Preset Buttons */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {PRESETS.map((p) => {
          const active = Math.abs(leadTime - p.hours) < 0.1;
          return (
            <button
              key={p.hours}
              onClick={() => {
                setIsPlaying(false);
                onChangeLeadTime(p.hours);
              }}
              className={`px-2.5 py-2 rounded-xl text-left transition-all border ${
                active
                  ? 'bg-gradient-to-r from-blue-600/30 to-cyan-600/30 border-cyan-400/80 shadow-md shadow-cyan-500/20 text-white'
                  : 'bg-slate-800/50 hover:bg-slate-800 border-slate-700/50 text-slate-400 hover:text-slate-200'
              }`}
            >
              <div className="text-xs font-bold font-mono text-cyan-300">{p.label}</div>
              <div className="text-[10px] text-slate-400 truncate">{p.tag}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
