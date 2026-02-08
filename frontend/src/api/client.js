export async function analyzeVideo(blob) {
  const form = new FormData();
  form.append("video", blob, "clip.webm");

  // If you set a Vite proxy (recommended), this works:
  const res = await fetch("/analyze", { method: "POST", body: form });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Analyze failed");
  }
  return res.json();
}
