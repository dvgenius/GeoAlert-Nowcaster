import React from 'react';
import { Zap, CloudRain, Waves, ShieldAlert } from 'lucide-react';

export default function HazardMetrics({ metrics }) {
  if (!metrics) return null;

  const hazards = [
    {
      id: 'cloudburst',
      title: 'Cloudburst Nowcast',
      sub: 'Extreme localized precipitation rate >100mm/h',
      icon: CloudRain,
      prob: metrics.cloudburst?.peak_probability ?? 0,
      status: metrics.cloudburst?.risk_status ?? 'MODERATE',
      color: '#ef4444',
      glow: 'shadow-red-500/20',
      barColor: 'from-orange-500 to-red-500'
    },
    {
      id: 'flash_flood',
      title: 'Flash Flood Risk',
      sub: 'Himalayan valley convergence & slope inundation',
      icon: Waves,
      prob: metrics.flash_flood?.peak_probability ?? 0,
      status: metrics.flash_flood?.risk_status ?? 'MODERATE',
      color: '#f97316',
      glow: 'shadow-orange-500/20',
      barColor: 'from-amber-500 to-orange-500'
    },
    {
      id: 'thunderstorm',
      title: 'Severe Thunderstorm',
      sub: 'Lightning strikes & severe microburst downdrafts',
      icon: Zap,
      prob: metrics.thunderstorm?.peak_probability ?? 0,
      status: metrics.thunderstorm?.risk_status ?? 'MODERATE',
      color: '#eab308',
      glow: 'shadow-yellow-500/20',
      barColor: 'from-yellow-400 to-amber-500'
    },
  ];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-white font-heading flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-cyan-400" />
          Multi-Task Hazard Probabilities
        </h3>
        <span className="text-[11px] font-mono text-slate-400">Sigmoid Outputs</span>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {hazards.map((h) => {
          const Icon = h.icon;
          const pct = Math.round(h.prob * 100);
          const isHigh = h.status === 'CRITICAL' || h.status === 'HIGH';

          return (
            <div
              key={h.id}
              className={`p-3.5 rounded-xl border border-slate-700/60 bg-slate-900/60 transition-all hover:border-slate-600 ${h.glow}`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-lg bg-slate-800 text-slate-200" style={{ color: h.color }}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-sm font-semibold text-white">{h.title}</h4>
                    <p className="text-[11px] text-slate-400">{h.sub}</p>
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-base font-bold font-mono text-white">{pct}%</span>
                  <div className="text-[10px] font-semibold" style={{ color: h.color }}>
                    {h.status}
                  </div>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden mt-2">
                <div
                  className={`h-full rounded-full bg-gradient-to-r ${h.barColor} transition-all duration-500`}
                  style={{ width: `${Math.max(5, pct)}%` }}
                ></div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
