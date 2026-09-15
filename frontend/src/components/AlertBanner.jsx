import React from 'react';
import { AlertTriangle, ShieldCheck, Siren, Clock, BellRing, Download } from 'lucide-react';

export default function AlertBanner({ summary, onExportGeoJSON }) {
  if (!summary) return null;

  const isRed = summary.alert_category === 'RED';
  const isOrange = summary.alert_category === 'ORANGE';
  const isGreen = summary.alert_category === 'GREEN';

  const bgGradient = isRed
    ? 'from-red-950/90 via-red-900/50 to-slate-900/90 border-red-500/50 shadow-red-900/20'
    : isOrange
    ? 'from-orange-950/90 via-orange-900/50 to-slate-900/90 border-orange-500/50 shadow-orange-900/20'
    : 'from-emerald-950/80 via-emerald-900/40 to-slate-900/90 border-emerald-500/40 shadow-emerald-900/20';

  const badgeClass = isRed
    ? 'bg-red-500 text-white animate-pulse'
    : isOrange
    ? 'bg-orange-500 text-white'
    : 'bg-emerald-500 text-white';

  return (
    <div className={`border rounded-2xl p-5 mb-5 bg-gradient-to-r ${bgGradient} shadow-xl backdrop-blur-md transition-all duration-300`}>
      <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
        {/* Left: Icon & Alert Statement */}
        <div className="flex items-start gap-4">
          <div className={`p-3 rounded-xl flex items-center justify-center ${isRed ? 'bg-red-500/20 text-red-400 pulse-red' : isOrange ? 'bg-orange-500/20 text-orange-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
            {isRed ? <Siren className="w-8 h-8" /> : isOrange ? <AlertTriangle className="w-8 h-8" /> : <ShieldCheck className="w-8 h-8" />}
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2 mb-1">
              <span className={`px-2.5 py-0.5 rounded-full text-xs font-black tracking-wider uppercase ${badgeClass}`}>
                {summary.alert_category} ALERT
              </span>
              <h2 className="text-xl font-bold text-white tracking-tight font-heading">
                {summary.alert_title}
              </h2>
            </div>
            <p className="text-sm text-slate-300 max-w-3xl leading-relaxed">
              {summary.recommended_action}
            </p>
          </div>
        </div>

        {/* Right: Metrics & Emergency Triggers */}
        <div className="flex flex-wrap items-center gap-3 w-full lg:w-auto justify-end">
          <div className="px-3.5 py-2 rounded-xl bg-slate-900/80 border border-slate-700/60 text-right">
            <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold flex items-center gap-1 justify-end">
              <Clock className="w-3 h-3 text-cyan-400" /> Lead Time
            </div>
            <div className="text-lg font-bold text-cyan-400 font-mono">
              {summary.lead_time_hours} Hours
            </div>
          </div>

          <div className="px-3.5 py-2 rounded-xl bg-slate-900/80 border border-slate-700/60 text-right">
            <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">
              Peak Hazard Risk
            </div>
            <div className={`text-lg font-bold font-mono ${isRed ? 'text-red-400' : isOrange ? 'text-orange-400' : 'text-emerald-400'}`}>
              {(summary.overall_peak_risk * 100).toFixed(1)}%
            </div>
          </div>

          <button
            onClick={onExportGeoJSON}
            className="flex items-center gap-2 px-3.5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-600/70 text-xs font-semibold transition-all hover:shadow-lg active:scale-95"
            title="Download GeoJSON Polygons"
          >
            <Download className="w-4 h-4 text-cyan-400" />
            <span>Export GeoJSON</span>
          </button>
        </div>
      </div>
    </div>
  );
}
