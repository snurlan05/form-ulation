export async function analyzeVideo(blob) {
  const form = new FormData();
  form.append("video", blob, "clip.webm");

  // Vite proxies /analyze to the FastAPI backend on localhost:8000
  // (see vite.config.js). In production, point this at the deployed API.
  const res = await fetch("/analyze", { method: "POST", body: form });

  if (!res.ok) {
    // FastAPI error bodies are JSON: {"detail": "..."}. Surface the
    // detail message instead of the raw JSON string.
    let message = "Analyze failed";
    try {
      const data = await res.json();
      if (data && data.detail) message = data.detail;
    } catch {
      message = (await res.text()) || message;
    }
    throw new Error(message);
  }
  return res.json();
}
