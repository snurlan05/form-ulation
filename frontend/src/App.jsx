import { useState } from "react";
import Recorder from "./components/Recorder";
import ChatPanel from "./components/ChatPanel";
import VideoLinks from "./components/VideoLinks";
import ScorePanel from "./components/ScorePanel";
import { analyzeVideo } from "./api/client";
import "./styles/app.css";

export default function App() {
  const [videoBlob, setVideoBlob] = useState(null);
  const [loading, setLoading] = useState(false);

  const [messages, setMessages] = useState([]);
  const [links, setLinks] = useState([]);
  const [score, setScore] = useState(null);
  const [breakdown, setBreakdown] = useState(null);

  async function onAnalyze() {
    if (!videoBlob) return;

    setLoading(true);
    setMessages([{ role: "user", text: "Analyze my set." }]);
    setLinks([]);
    setScore(null);
    setBreakdown(null);

    try {
      const data = await analyzeVideo(videoBlob);

      if (data.score !== undefined) setScore(data.score);
      if (data.breakdown) setBreakdown(data.breakdown);

      setMessages((prev) => [...prev, ...(data.messages || [])]);
      setLinks(data.links || []);
    } catch (e) {
      setMessages((prev) => [...prev, { role: "coach", text: `Error: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <div className="header">
        <div className="title">Form-ulation</div>
        <div className="muted">Record → Analyze → Coaching + videos</div>
      </div>

      <div className="grid">
        <div className="left">
          <Recorder onRecorded={setVideoBlob} />
          <button className="btn primary" onClick={onAnalyze} disabled={!videoBlob || loading}>
            {loading ? "Analyzing..." : "Analyze"}
          </button>
          <div className="muted">Tip: keep upper body fully in frame + good lighting.</div>
        </div>

        <div className="right">
          <ScorePanel score={score} breakdown={breakdown} />
          <ChatPanel messages={messages} />
          <VideoLinks links={links} />
        </div>
      </div>
    </div>
  );
}
