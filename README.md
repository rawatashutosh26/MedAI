<div align="center">

![MedAI Pro](docs/images/medai-banner.svg)

**AI-assisted medical imaging and ICU sepsis risk assessment**

| | |
| :--- | :--- |
| **Live app** | `https://YOUR_USERNAME-medai.hf.space` ← *replace with your Hugging Face Space URL* |
| **Repository** | `https://github.com/YOUR_USERNAME/YOUR_REPO` ← *replace with your GitHub repo URL* |

</div>

---

## Overview

MedAI Pro is a full-stack diagnostic assistant with:

- **Imaging modules:** Chest X-Ray, Brain MRI, Retinal scan, Skin lesion analysis  
- **Clinical module:** Sepsis risk from vitals and labs (LSTM + XGBoost + Random Forest ensemble)  
- **Auth & history:** Express + JWT cookies; optional MongoDB Atlas for users and reports  
- **Patient history (UI):** Local browser storage for recent analyses with thumbnails  

> **Disclaimer:** For research and education. Not a medical device and not a substitute for professional clinical judgment.

---

## Screenshots & diagrams

<p align="center">
  <img src="docs/images/medai-banner.svg" alt="MedAI Pro banner" width="100%" />
</p>

<p align="center">
  <img src="docs/images/architecture.svg" alt="Architecture diagram" width="100%" />
</p>

*Optional:* Add real UI screenshots under `docs/images/` (e.g. `dashboard.png`, `sepsis.png`) and link them here:

```markdown
![Dashboard](docs/images/dashboard.png)
![Sepsis module](docs/images/sepsis.png)
```

---

## Tech stack

| Layer | Stack |
| :--- | :--- |
| Frontend | React 19, Vite, Tailwind CSS, Framer Motion, Axios |
| API gateway | Node.js, Express, Mongoose, JWT, Helmet, CORS |
| ML API | Python, FastAPI, Uvicorn |
| ML | TensorFlow/Keras, PyTorch, scikit-learn, XGBoost, OpenCV |
| Data | MongoDB Atlas (optional), `gdown` for model files from Google Drive |
| Deploy | Docker (multi-stage), Hugging Face Spaces (example) |

---

## Project structure

```
├── api.py              # FastAPI app, model load, /predict/* routes
├── requirements.txt    # Python dependencies
├── Dockerfile          # Production image (frontend build + Node + Python)
├── start.sh            # Starts FastAPI then Express on port 7860 (HF Spaces)
├── modules/            # Python helpers used by api.py
├── backend/            # Express server (auth, proxy to Python, static SPA)
├── frontend/           # React SPA
└── docs/images/        # README assets (SVG banner, architecture)
```

Large model files are **not** committed; they are downloaded at startup from Google Drive (see `api.py` `MODEL_FILES`).

---

## Prerequisites

- **Node.js** 20+  
- **Python** 3.10+  
- **MongoDB Atlas** URI (optional; without it, dev uses in-memory auth)  

---

## Local development

### 1. Python (FastAPI)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### 2. Node backend (Express)

```bash
cd backend
npm install
# Create backend/.env with MONGO_URI, JWT_SECRET, PYTHON_API_URL=http://127.0.0.1:8000
node server.js
```

### 3. Frontend (Vite)

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` to `http://localhost:5000` (see `frontend/vite.config.js`). The frontend uses `baseURL: /api` so cookies and routes work in dev.

---

## Environment variables

**Backend (`backend/.env` or hosting secrets)**

| Variable | Example | Purpose |
| :--- | :--- | :--- |
| `PORT` | `5000` | Express port (local) |
| `PYTHON_API_URL` | `http://127.0.0.1:8000` | FastAPI base URL |
| `MONGO_URI` | `mongodb+srv://...` | Atlas (optional) |
| `JWT_SECRET` | long random string | JWT signing |
| `NODE_ENV` | `production` | Static SPA + stricter CORS/cookies |
| `FRONTEND_ORIGIN` | `https://your-frontend.com` | Allowed CORS origin in production |

**Frontend (production build only if split from backend)**

| Variable | Purpose |
| :--- | :--- |
| `VITE_API_BASE` | Full API base, e.g. `https://your-api.com/api` |

**Docker / Hugging Face:** set secrets in the Space dashboard; `start.sh` sets `PYTHON_API_URL=http://127.0.0.1:8000` and `PORT=7860`.

---

## Docker

```bash
docker build -t medai .
docker run -p 7860:7860 -e MONGO_URI="mongodb+srv://..." -e JWT_SECRET="..." medai
```

Open `http://localhost:7860`.

---

## Deploy links (fill in for your README)

Copy the table at the top of this file and replace:

1. **Live app** — your Hugging Face Space URL (or Render / other host).  
2. **Repository** — your GitHub repo URL.  

If you use **Vercel** for frontend only, add another row:

```markdown
| **Frontend (Vercel)** | `https://your-app.vercel.app` |
```

---

## License & citation

Use your chosen license in a `LICENSE` file. If this is coursework, cite your institution and capstone requirements as needed.

---

*Generated for MedAI Pro — paste this entire file into GitHub as `README.md` in the repository root.*
