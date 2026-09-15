# GeoAlert Nowcaster — Hyper-Local Severe Weather Nowcasting
**Team ID:** KT-2132 | **Team Name:** NowCast  
**Event:** Hackathon Software (Submission: September 2026)

## Overview
GeoAlert Nowcaster is an AI-powered early warning system designed to predict severe thunderstorms, cloudbursts, and flash floods simultaneously with a 2 to 6-hour actionable lead time and sub-200 ms inference latency.

## Key Features
- **Shared Multi-Task Backbone:** ConvLSTM U-Net predicting thunderstorms, cloudbursts, and flash floods concurrently.
- **PINN Thermodynamics:** Physics-informed regularizer tracking Integrated Water Vapor (IWV) and Cloud Top Temperature cooling rate (-dCTT/dt).
- **DEM Terrain Routing:** CartoDEM slope and drainage convolution for catchment flash-flood modeling.
- **Explainable AI:** Captum Integrated Gradients attribution + automated thermodynamic text narratives.
- **Disaster Response SOPs:** Auto-generated district evacuation guidance and 1-click GeoJSON export for state GIS integration.

## Tech Stack
- **Deep Learning / XAI:** PyTorch, ConvLSTM, U-Net, Captum, PINN
- **Backend & APIs:** FastAPI, Uvicorn, NumPy, SciPy, Rasterio
- **Frontend & GIS:** React / Vite, Tailwind CSS, Leaflet / MapLibre GL
- **Data Baselines:** INSAT-3DR (MOSDAC), IMDAA Reanalysis, CartoDEM

## Setup & Local Run
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd ../frontend
npm install
npm run dev
