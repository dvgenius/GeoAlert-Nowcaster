import React, { useState, useEffect, useCallback } from 'react';
import Navbar from './components/Navbar';
import AlertBanner from './components/AlertBanner';
import LeadTimeControl from './components/LeadTimeControl';
import MapViewer from './components/MapViewer';
import HazardMetrics from './components/HazardMetrics';
import XaiDrawer from './components/XaiDrawer';
import ActionProtocols from './components/ActionProtocols';
import { RefreshCw, Activity, Database, Satellite, Server, Compass } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || "https://geoalert-nowcaster.onrender.com";

export default function App() {
  const [region, setRegion] = useState('uttarakhand');
  const [leadTime, setLeadTime] = useState(3.0);
  const [nowcastData, setNowcastData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [backendConnected, setBackendConnected] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // Fetch prediction data from FastAPI
  const fetchNowcast = useCallback(async (hours, selectedRegion) => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const res = await fetch(
        `${API_BASE}/api/v1/predict/nowcast?lead_time_hours=${hours}&region=${selectedRegion}`
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to fetch nowcast`);
      const data = await res.json();
      setNowcastData(data);
      setBackendConnected(true);
    } catch (err) {
      console.warn('API fetch failed:', err);
      setBackendConnected(false);
      setErrorMsg(err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // On lead-time or region change
  useEffect(() => {
    fetchNowcast(leadTime, region);
  }, [leadTime, region, fetchNowcast]);

  // Export GeoJSON function
  const handleExportGeoJSON = () => {
    if (!nowcastData || !nowcastData.geojson) return;
    const blob = new Blob([JSON.stringify(nowcastData.geojson, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${region}_nowcast_T${leadTime}h_${Date.now()}.geojson`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const regionMeta = nowcastData?.region_info;

  return (
    <div className="min-h-screen flex flex-col bg-[#070a12] text-slate-100 selection:bg-cyan-500 selection:text-slate-900">
      {/* Header with Region Selector */}
      <Navbar
        backendConnected={backendConnected}
        region={region}
        onChangeRegion={setRegion}
        regionInfo={regionMeta}
      />

      {/* Main Dashboard Container */}
      <main className="flex-1 max-w-[1700px] w-full mx-auto p-4 md:p-6 space-y-5">
        {/* Connection Notice if disconnected */}
        {!backendConnected && (
          <div className="p-3 rounded-xl bg-amber-500/15 border border-amber-500/40 text-amber-300 text-xs flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4 text-amber-400" />
              <span>
                Connecting to FastAPI backend at <code className="font-mono bg-amber-950/60 px-1.5 py-0.5 rounded">{API_BASE}</code>...
              </span>
            </div>
            <button
              onClick={() => fetchNowcast(leadTime, region)}
              className="px-2.5 py-1 rounded bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 text-xs font-semibold flex items-center gap-1"
            >
              <RefreshCw className="w-3 h-3" /> Retry
            </button>
          </div>
        )}

        {/* Emergency Alert Banner */}
        {nowcastData?.summary && (
          <AlertBanner
            summary={nowcastData.summary}
            onExportGeoJSON={handleExportGeoJSON}
          />
        )}

        {/* 2-Column Responsive Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
          {/* Left Column (8 of 12 cols): Controls & Map */}
          <div className="lg:col-span-8 space-y-4">
            {/* Lead Time Slider */}
            <LeadTimeControl
              leadTime={leadTime}
              onChangeLeadTime={setLeadTime}
              isLoading={isLoading}
            />

            {/* Interactive Cartographic Map */}
            <MapViewer
              geojson={nowcastData?.geojson}
              leadTimeHours={leadTime}
              alertCategory={nowcastData?.summary?.alert_category}
              regionInfo={regionMeta}
            />

            {/* Sensor Telemetry Stats Strip */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="glass-panel p-3 rounded-xl border border-slate-800 text-xs">
                <div className="text-slate-400 flex items-center gap-1 mb-1">
                  <Satellite className="w-3.5 h-3.5 text-cyan-400" /> INSAT-3DR Telemetry
                </div>
                <div className="font-mono font-bold text-white text-sm">Channel 4 (TIR-1)</div>
                <div className="text-[10px] text-emerald-400">15-min Rapid Scan</div>
              </div>

              <div className="glass-panel p-3 rounded-xl border border-slate-800 text-xs">
                <div className="text-slate-400 flex items-center gap-1 mb-1">
                  <Compass className="w-3.5 h-3.5 text-indigo-400" /> Bounding Box
                </div>
                <div className="font-mono font-bold text-white text-xs truncate">
                  {regionMeta?.bounding_box?.min_lat}°–{regionMeta?.bounding_box?.max_lat}°N
                </div>
                <div className="text-[10px] text-cyan-400 font-mono">
                  {regionMeta?.lat_step_deg}° grid step
                </div>
              </div>

              <div className="glass-panel p-3 rounded-xl border border-slate-800 text-xs">
                <div className="text-slate-400 flex items-center gap-1 mb-1">
                  <Activity className="w-3.5 h-3.5 text-amber-400" /> Physics Regularizer
                </div>
                <div className="font-mono font-bold text-white text-sm">PINN Thermodynamics</div>
                <div className="text-[10px] text-cyan-400">IWV & -dCTT/dt Envelope</div>
              </div>

              <div className="glass-panel p-3 rounded-xl border border-slate-800 text-xs">
                <div className="text-slate-400 flex items-center gap-1 mb-1">
                  <RefreshCw className={`w-3.5 h-3.5 text-cyan-400 ${isLoading ? 'animate-spin' : ''}`} /> Latency
                </div>
                <div className="font-mono font-bold text-white text-sm">&lt; 190 ms</div>
                <div className="text-[10px] text-emerald-400">Real-time Inference</div>
              </div>
            </div>
          </div>

          {/* Right Column (4 of 12 cols): Multi-Hazard Heads, XAI Drawer, Protocols */}
          <div className="lg:col-span-4 space-y-4">
            {/* Multi-Hazard Risk Meters */}
            <div className="glass-panel p-4 rounded-2xl border border-slate-700/60 shadow-lg">
              <HazardMetrics metrics={nowcastData?.hazard_metrics} />
            </div>

            {/* Captum XAI Feature Importance Drawer */}
            <XaiDrawer xai={nowcastData?.xai_attribution} />

            {/* Emergency Action Checklist & Contacts */}
            <ActionProtocols alertCategory={nowcastData?.summary?.alert_category} />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-8 border-t border-slate-800/80 px-6 py-4 text-xs text-slate-500 text-center glass-panel">
        <p>
          <b className="text-slate-400">GeoAlert Nowcaster:</b> AI-Driven Hyper-Local Early Warning System for Severe Weather Nowcasting.
          Pan-India Coverage: Uttarakhand, Mumbai, Western Ghats, North-East. Powered by PyTorch ConvLSTM U-Net, Captum Integrated Gradients, and FastAPI GeoJSON Streaming.
        </p>
      </footer>
    </div>
  );
}
