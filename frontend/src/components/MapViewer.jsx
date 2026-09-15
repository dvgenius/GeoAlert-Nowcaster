import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { Layers, MapPin, ZoomIn, ZoomOut, Maximize2, Compass } from 'lucide-react';

const REGION_LANDMARKS = {
  uttarakhand: [
    { name: 'Joshimath (Urban Hub)', coords: [30.556, 79.566], elev: '1,890m' },
    { name: 'Tapovan Gorge (Rishi Ganga)', coords: [30.490, 79.630], elev: '2,040m' },
    { name: 'Badrinath Gateway', coords: [30.744, 79.493], elev: '3,133m' },
    { name: 'Pipalkoti (Relocation Camp)', coords: [30.428, 79.428], elev: '1,340m' },
  ],
  mumbai: [
    { name: 'Mithi River Outfall', coords: [19.060, 72.845], elev: '5m' },
    { name: 'Kurla Junction (Flood Hotspot)', coords: [19.068, 72.880], elev: '8m' },
    { name: 'Sanjay Gandhi NP Ridges', coords: [19.220, 72.910], elev: '420m' },
    { name: 'Dadar Inundation Corridor', coords: [19.020, 72.840], elev: '4m' },
  ],
  western_ghats: [
    { name: 'Meppadi / Chooralmala (Landslide Core)', coords: [11.550, 76.120], elev: '940m' },
    { name: 'Vythiri Hill Crest', coords: [11.550, 76.040], elev: '1,300m' },
    { name: 'Chaliyar River Funnel', coords: [11.450, 76.150], elev: '420m' },
    { name: 'Kalpetta District HQ', coords: [11.610, 76.080], elev: '780m' },
  ],
  northeast: [
    { name: 'Sohra / Cherrapunji Plateau', coords: [25.270, 91.730], elev: '1,430m' },
    { name: 'Nohkalikai Gorge Head', coords: [25.280, 91.680], elev: '1,350m' },
    { name: 'Shella River Valley', coords: [25.180, 91.640], elev: '180m' },
    { name: 'Mawsynram Crest', coords: [25.300, 91.580], elev: '1,400m' },
  ],
};

const CARTO_API_KEY = import.meta.env.VITE_CARTO_API_KEY || 'cb1_3jek_1_02d7ed44dfddb03303c39d47';

export default function MapViewer({ geojson, leadTimeHours, alertCategory, regionInfo }) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const geojsonLayerRef = useRef(null);
  const landmarksLayerRef = useRef(null);

  const [basemap, setBasemap] = useState('dark');
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [totalAlertCells, setTotalAlertCells] = useState(0);

  const centerLat = regionInfo?.center?.lat ?? 30.45;
  const centerLon = regionInfo?.center?.lon ?? 79.55;
  const currentPreset = regionInfo?.preset ?? 'uttarakhand';

  // 1. Initialize Leaflet Map
  useEffect(() => {
    if (!mapContainerRef.current) return;
    if (mapInstanceRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [centerLat, centerLon],
      zoom: 10,
      zoomControl: false,
      attributionControl: false,
      minZoom: 6,
      maxZoom: 15,
    });

    const darkTiles = L.tileLayer(
      `https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png?key=${CARTO_API_KEY}`,
      {
        subdomains: 'abcd',
        maxZoom: 19,
        attribution: '&copy; <a href="https://carto.com/">CARTO</a>'
      }
    ).addTo(map);

    const voyagerTiles = L.tileLayer(
      `https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png?key=${CARTO_API_KEY}`,
      {
        subdomains: 'abcd',
        maxZoom: 19,
        attribution: '&copy; <a href="https://carto.com/">CARTO</a>'
      }
    );

    const satelliteTiles = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 18 }
    );

    const landmarkGroup = L.layerGroup().addTo(map);
    landmarksLayerRef.current = landmarkGroup;

    mapInstanceRef.current = {
      map,
      darkTiles,
      voyagerTiles,
      satelliteTiles,
    };

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // 2. Pan/Fly to New Region Center when Region Changes
  useEffect(() => {
    if (!mapInstanceRef.current || !regionInfo?.center) return;
    const { map } = mapInstanceRef.current;
    map.flyTo([centerLat, centerLon], 10, { duration: 1.2 });

    // Update Landmarks Layer
    if (landmarksLayerRef.current) {
      landmarksLayerRef.current.clearLayers();
      const landmarkList = REGION_LANDMARKS[currentPreset] || [];
      landmarkList.forEach((lm) => {
        const icon = L.divIcon({
          className: 'custom-landmark-icon',
          html: `<div style="background: rgba(15,23,42,0.9); border: 1px solid #38bdf8; border-radius: 6px; padding: 2px 6px; font-size: 10px; font-weight: 700; color: #38bdf8; white-space: nowrap; box-shadow: 0 2px 6px rgba(0,0,0,0.6); display: flex; align-items: center; gap: 4px;">
                  <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #38bdf8;"></span>
                  ${lm.name}
                 </div>`,
          iconSize: [110, 24],
          iconAnchor: [55, 12]
        });

        L.marker(lm.coords, { icon })
          .bindTooltip(`<b>${lm.name}</b><br/>Elevation: ${lm.elev}<br/>Coords: ${lm.coords[0].toFixed(3)}°N, ${lm.coords[1].toFixed(3)}°E`, {
            direction: 'top',
            className: 'leaflet-tooltip-dark'
          })
          .addTo(landmarksLayerRef.current);
      });
    }
  }, [centerLat, centerLon, currentPreset]);

  // 3. Basemap Switcher
  useEffect(() => {
    if (!mapInstanceRef.current) return;
    const { map, darkTiles, voyagerTiles, satelliteTiles } = mapInstanceRef.current;

    // Remove all layers first
    map.removeLayer(darkTiles);
    map.removeLayer(voyagerTiles);
    map.removeLayer(satelliteTiles);

    if (basemap === 'satellite') {
      satelliteTiles.addTo(map);
    } else if (basemap === 'voyager') {
      voyagerTiles.addTo(map);
    } else {
      darkTiles.addTo(map);
    }
  }, [basemap]);

  // 4. Render GeoJSON Risk Polygons
  useEffect(() => {
    if (!mapInstanceRef.current || !geojson) return;
    const { map } = mapInstanceRef.current;

    if (geojsonLayerRef.current) {
      map.removeLayer(geojsonLayerRef.current);
      geojsonLayerRef.current = null;
    }

    if (geojson.features) {
      setTotalAlertCells(geojson.features.length);
    }

    const styleFeature = (feature) => {
      const score = feature.properties?.risk_score ?? 0.5;
      const isCritical = score >= 0.70;
      const isHigh = score >= 0.55;

      const fillColor = isCritical ? '#ef4444' : isHigh ? '#f97316' : '#eab308';
      const strokeColor = isCritical ? '#fca5a5' : isHigh ? '#fdba74' : '#fde047';

      return {
        fillColor: fillColor,
        weight: isCritical ? 2.2 : 1.4,
        opacity: 0.95,
        color: strokeColor,
        fillOpacity: isCritical ? 0.65 : isHigh ? 0.50 : 0.35,
        dashArray: isCritical ? null : '2, 3',
      };
    };

    const onEachFeature = (feature, layer) => {
      const props = feature.properties || {};

      layer.on({
        mouseover: (e) => {
          const l = e.target;
          l.setStyle({
            weight: 3.5,
            color: '#38bdf8',
            fillOpacity: 0.85,
          });
          l.bringToFront();
        },
        mouseout: (e) => {
          if (geojsonLayerRef.current) {
            geojsonLayerRef.current.resetStyle(e.target);
          }
        },
        click: () => {
          setSelectedFeature(props);
        },
      });

      const hazardTitle = (props.hazard_type || 'Convective Hazard').toUpperCase();
      const popupHtml = `
        <div style="font-family: 'Inter', sans-serif; min-width: 220px; font-size: 12px;">
          <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 6px; margin-bottom: 6px;">
            <span style="font-weight: 800; color: ${props.color || '#ef4444'}; text-transform: uppercase;">
              ${hazardTitle}
            </span>
            <span style="background: ${props.color || '#ef4444'}; color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700;">
              ${props.risk_level || 'ALERT'}
            </span>
          </div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; margin-bottom: 6px; font-size: 11px;">
            <div><span style="color: #94a3b8;">Risk Score:</span> <b style="color: #fff;">${(props.risk_score * 100).toFixed(1)}%</b></div>
            <div><span style="color: #94a3b8;">Elevation:</span> <b style="color: #fff;">${props.elevation_m}m</b></div>
            <div><span style="color: #94a3b8;">Latitude:</span> <b style="color: #fff;">${props.center_lat}°N</b></div>
            <div><span style="color: #94a3b8;">Longitude:</span> <b style="color: #fff;">${props.center_lon}°E</b></div>
          </div>
          <div style="background: rgba(0,0,0,0.3); padding: 6px; border-radius: 6px; border-left: 3px solid ${props.color || '#ef4444'}; font-size: 11px; color: #e2e8f0; line-height: 1.35;">
            <b>Action Protocol:</b> ${props.recommended_action}
          </div>
        </div>
      `;
      layer.bindPopup(popupHtml);
    };

    const newGeojsonLayer = L.geoJSON(geojson, {
      style: styleFeature,
      onEachFeature: onEachFeature,
    }).addTo(map);

    geojsonLayerRef.current = newGeojsonLayer;
  }, [geojson]);

  const handleZoomIn = () => mapInstanceRef.current?.map.zoomIn();
  const handleZoomOut = () => mapInstanceRef.current?.map.zoomOut();
  const handleResetView = () => mapInstanceRef.current?.map.setView([centerLat, centerLon], 10);

  return (
    <div className="relative w-full h-full min-h-[540px] rounded-2xl overflow-hidden border border-slate-700/60 shadow-2xl glass-panel">
      <div ref={mapContainerRef} className="w-full h-full" style={{ minHeight: '540px' }} />

      {/* Floating HUD Controls (Top Right) */}
      <div className="absolute top-4 right-4 z-[400] flex flex-col gap-2">
        <div className="glass-panel rounded-xl p-1 border border-slate-700/80 shadow-lg flex gap-1">
          <button
            onClick={() => setBasemap('dark')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              basemap === 'dark'
                ? 'bg-cyan-500 text-slate-950 shadow-md font-bold'
                : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            Dark Matter
          </button>
          <button
            onClick={() => setBasemap('voyager')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              basemap === 'voyager'
                ? 'bg-cyan-500 text-slate-950 shadow-md font-bold'
                : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            Carto Voyager
          </button>
          <button
            onClick={() => setBasemap('satellite')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              basemap === 'satellite'
                ? 'bg-cyan-500 text-slate-950 shadow-md font-bold'
                : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            Topo Satellite
          </button>
        </div>

        <div className="glass-panel rounded-xl p-1 border border-slate-700/80 shadow-lg flex flex-col gap-1 w-9 items-center">
          <button
            onClick={handleZoomIn}
            className="p-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/80"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={handleZoomOut}
            className="p-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/80"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <div className="w-full h-[1px] bg-slate-700/60 my-0.5"></div>
          <button
            onClick={handleResetView}
            className="p-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800/80"
            title="Center Current Region"
          >
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Floating Status Badge (Top Left) */}
      <div className="absolute top-4 left-4 z-[400] pointer-events-none">
        <div className="glass-panel px-3.5 py-2 rounded-xl border border-slate-700/80 shadow-lg pointer-events-auto flex items-center gap-2.5 text-xs">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
          </span>
          <div>
            <div className="font-bold text-white flex items-center gap-1.5">
              <span>{regionInfo?.label || 'Regional Risk Overlay'}</span>
              <span className="font-mono text-cyan-400 font-semibold">T+{leadTimeHours}h</span>
            </div>
            <div className="text-[11px] text-slate-400 font-mono">
              {totalAlertCells} high-risk convective cells identified
            </div>
          </div>
        </div>
      </div>

      {/* Floating Legend (Bottom Left) */}
      <div className="absolute bottom-4 left-4 z-[400] glass-panel px-4 py-3 rounded-xl border border-slate-700/80 shadow-xl text-xs space-y-1.5">
        <div className="font-bold text-white text-[11px] tracking-wider uppercase mb-1 flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-cyan-400" />
          Hazard Severity Index
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded bg-red-500 border border-red-300 shadow-sm shadow-red-500/50"></span>
          <span className="text-slate-200 font-semibold">Critical Risk (&gt;70%)</span>
          <span className="text-[10px] text-slate-400">— Mandatory Evac</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded bg-orange-500 border border-orange-300"></span>
          <span className="text-slate-200 font-semibold">High Risk (55–70%)</span>
          <span className="text-[10px] text-slate-400">— Standby</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded bg-yellow-500 border border-yellow-300"></span>
          <span className="text-slate-200 font-semibold">Moderate (40–55%)</span>
          <span className="text-[10px] text-slate-400">— Advisory</span>
        </div>
      </div>

      {/* Selected Cell Floating Card (Bottom Right) */}
      {selectedFeature && (
        <div className="absolute bottom-4 right-4 z-[400] glass-panel p-3.5 rounded-xl border border-cyan-500/40 shadow-2xl max-w-xs text-xs">
          <div className="flex items-center justify-between border-b border-slate-700 pb-1.5 mb-2">
            <span className="font-bold text-cyan-400 uppercase tracking-wide">
              Selected Cell Telemetry
            </span>
            <button
              onClick={() => setSelectedFeature(null)}
              className="text-slate-400 hover:text-white px-1 font-bold"
            >
              ✕
            </button>
          </div>
          <div className="space-y-1 font-mono text-[11px] text-slate-300">
            <div>Hazard: <b className="text-white uppercase">{selectedFeature.hazard_type}</b></div>
            <div>Risk Score: <b className="text-cyan-300">{(selectedFeature.risk_score * 100).toFixed(1)}%</b></div>
            <div>Elevation: <b>{selectedFeature.elevation_m}m</b></div>
            <div>Location: <b>{selectedFeature.center_lat}°N, {selectedFeature.center_lon}°E</b></div>
          </div>
          <div className="mt-2 text-[10px] text-slate-300 bg-slate-800/80 p-2 rounded border border-slate-700">
            {selectedFeature.recommended_action}
          </div>
        </div>
      )}
    </div>
  );
}
