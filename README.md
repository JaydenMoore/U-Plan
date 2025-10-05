# 🌍 U-Plan — NASA-Powered Risk Insights

A lightweight web app that helps **urban planners** assess environmental risks and construction conditions in any location using **NASA's open APIs**.

Built for **NASA Space Apps Challenge 2025**.

---

## 🚀 Overview

U-Plan lets you **click on any location on a map** and instantly see:
- 🌧️ Average rainfall and potential **flood risk**
- 🌡️ Average temperature and **heat exposure**
- 🔥 Optional: Nearby fire data (via NASA FIRMS)
- 🧱 Quick summary of area suitability (AI-generated)

All data is fetched in real-time from **NASA APIs**, so **no database** is required.

---

## 🧠 Tech Stack

| Layer | Tech |
|-------|------|
| **Frontend** | React (Vite) + Tailwind CSS + Mapbox GL JS |
| **Backend** | FastAPI (Python) |
| **Data Source** | NASA POWER API (climate), NASA FIRMS (fires) |
| **AI Summary (optional)** | OpenAI API or local MLX model |
| **Hosting** | Frontend → Vercel, Backend → Render |

---

## 🧩 Architecture

```
Frontend (React + Mapbox)
↓
Backend (FastAPI)
↓
NASA APIs (POWER, FIRMS)
↓
AI Summarizer (optional)
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/urban-planner-ai.git
cd urban-planner-ai
```

---

### 2. Set up the **backend**

#### 🐍 Install dependencies

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 📦 Example `requirements.txt`

```text
fastapi
uvicorn
requests
```

#### ▶️ Run FastAPI

```bash
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000`

---

### 3. Set up the **frontend**

#### 💻 Install dependencies

```bash
cd frontend
npm install
```

#### 🔑 Environment variables

Create `.env` file in `/frontend` with your Mapbox token:

```bash
VITE_MAPBOX_TOKEN=pk.eyJ1IjoieW91cm5hbWUiLCJhIjoiY2xjZ2FybHFlMDFlZzNrbW9ndWZ4NXhqNiJ9.abc123xyz
```

Create `.env` file in `/backend`
```bash
GOOGLE_API_KEY=AIzaSyArnO_iLwEAsTQkCPKoq3Atm-SxF57ggBk
SUPABASE_URL=https://cfxdoiwlxlibralnjrpr.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNmeGRvaXdseGxpYnJhbG5qcnByIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1OTY0NjUwOSwiZXhwIjoyMDc1MjIyNTA5fQ.CqF9o7EBGlYTcw4bViVsZsIB1ck2V3yI0_XOVK2OUIA
```

#### ▶️ Start the frontend

```bash
npm run dev
```

Frontend runs at `http://localhost:5173`

---

## 🧰 NASA API References

| Data                       | Source                                                           | Example Endpoint                                                                                                                                             |
| -------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Rainfall & Temperature** | [NASA POWER API](https://power.larc.nasa.gov/docs/services/api/) | `https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=PRECTOT,T2M&community=RE&longitude=-95.36&latitude=29.76&start=2020&end=2024&format=JSON` |
| **Fires (optional)**       | [NASA FIRMS API](https://firms.modaps.eosdis.nasa.gov/api/)      | `https://firms.modaps.eosdis.nasa.gov/api/area/csv/<api_key>/VIIRS_SNPP_NRT/world/1`                                                                         |

> 🧠 You can register for a free [NASA Earthdata Login](https://urs.earthdata.nasa.gov/) if you need authenticated datasets.

---

## 🧮 Simple Risk Logic (MVP)

```python
flood_risk = "High" if rainfall > 150 else "Medium" if rainfall > 80 else "Low"
heat_risk = "High" if temp > 32 else "Medium" if temp > 25 else "Low"
```

---

## 🧠 AI Summary Prompt (optional)

> "Summarize this location: rainfall = {rainfall} mm/month, temperature = {temp}°C, flood risk = {flood}, heat risk = {heat}. Write a short urban planning note."

---

## 💾 Optional Add-ons

| Feature        | Description                                                      |
| -------------- | ---------------------------------------------------------------- |
| **Caching**    | Use `functools.lru_cache` or Redis to avoid refetching NASA data |
| **Elevation**  | Add OpenTopography or Google Elevation API                       |
| **AI Model**   | Replace OpenAI API with local MLX Phi-2 or TinyLlama             |
| **Deployment** | Deploy backend → Render, frontend → Vercel                       |

---

## 🧭 Example Folder Structure

```
urban-planner-ai/
├── backend/
│   ├── main.py
│   ├── nasa_utils.py
│   ├── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/Map.jsx
│   │   ├── components/ResultsCard.jsx
│   ├── index.html
│   ├── package.json
│   └── tailwind.config.js
└── README.md
```

---

## 🧪 Test Locally

1. Run backend (`uvicorn main:app --reload`)
2. Run frontend (`npm run dev`)
3. Open browser → click anywhere on the map
4. See NASA data + risk summary appear in sidebar

---

## 🚀 Deploy (optional)

### Frontend → [Vercel](https://vercel.com/)

```bash
vercel
```

### Backend → [Render](https://render.com/)

* New Web Service → Python → `uvicorn main:app`
* Expose port `8000`
* Add CORS middleware in `main.py`:

  ```python
  from fastapi.middleware.cors import CORSMiddleware
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```

---

## 🤖 Copilot Prompt

> "Create a React + FastAPI prototype called *U-Plan*. The user clicks on a Mapbox map to get latitude and longitude. The backend calls NASA POWER API to fetch rainfall and temperature data, calculates flood and heat risks, and returns them as JSON. The frontend displays the data in a styled card using Tailwind CSS."

---

## 🏁 MVP Goal

By the end of the hackathon:

* A user can click any location 🌎
* See NASA data (rainfall/temp)
* Read risk levels & a summary 🧾
* All live — no database, no manual data prep.

---

## 🛰️ Credits

* **NASA POWER API** – Global meteorological data
* **NASA FIRMS** – Fire detection data
* **Mapbox** – Interactive map visualization
* **FastAPI + React** – Modern full-stack framework

---

**Built with ❤️ for NASA Space Apps Challenge 2025**
