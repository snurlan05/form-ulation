# Form-ulation — React Frontend

React 19 + Vite web UI for Form-ulation. Record a bicep-curl clip with
your webcam, send it to the FastAPI backend (`POST /analyze`), and get
a form score, a feature breakdown, AI coach messages, and tutorial
video links.

## Run

```bash
# 1. Start the backend first (from the repo root)
pip install -r backend/requirements.txt
uvicorn backend.app:app --port 8000

# 2. In another terminal, start the frontend
cd frontend
npm install
npm run dev        # http://localhost:5173 — /analyze is proxied to :8000
npm run build      # production build (verified)
```

The backend must be running on `localhost:8000` — the Vite dev server
proxies `/analyze` there (see `vite.config.js`). Without it, pressing
**Analyze** shows a backend error in the Coach panel.

## How it works

| File | Purpose |
|---|---|
| `src/components/Recorder.jsx` | Webcam via `getUserMedia`, records a WebM clip, shows a replay preview |
| `src/components/ScorePanel.jsx` | Numeric form score (0–100) + feature breakdown |
| `src/components/ChatPanel.jsx` | Coach messages returned by the backend |
| `src/components/VideoLinks.jsx` | Tutorial video links returned by the backend |
| `src/api/client.js` | `POST /analyze` with the clip as `multipart/form-data` (field `video`) |

## API contract (`POST /analyze`)

Request: `multipart/form-data`, field `video` (WebM/MP4/MOV, max 100 MB).

Response (JSON):

```json
{
  "score": 82,
  "breakdown": { "Form score": "82 / 100", "Good-form frames": "148 / 180 (82%)" },
  "messages": [{ "role": "coach", "text": "Score 82/100 ..." }],
  "links": [{ "title": "...", "url": "https://www.youtube.com/watch?v=...", "tag": "..." }]
}
```

If no person is detected, the backend returns HTTP 422 with a
`{"detail": "..."}` message, which is shown in the Coach panel.

## Notes

- Recording needs camera permission and a secure context (`https://`
  or `localhost`). If access is denied, the Recorder shows an error
  instead of a blank camera box.
- `node_modules/` is intentionally not committed — run `npm install`.
