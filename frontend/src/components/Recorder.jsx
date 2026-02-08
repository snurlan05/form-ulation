import { useEffect, useRef, useState } from "react";

export default function Recorder({ onRecorded }) {
  const videoRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);

  const [stream, setStream] = useState(null);
  const [recording, setRecording] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  useEffect(() => {
    (async () => {
      const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      setStream(s);
      if (videoRef.current) videoRef.current.srcObject = s;
    })();

    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function start() {
    if (!stream) return;
    chunksRef.current = [];

    const options = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? { mimeType: "video/webm;codecs=vp9" }
      : { mimeType: "video/webm" };

    const mr = new MediaRecorder(stream, options);
    recorderRef.current = mr;

    mr.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
    };

    mr.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "video/webm" });
      const url = URL.createObjectURL(blob);
      setPreviewUrl(url);
      onRecorded(blob);
    };

    mr.start();
    setRecording(true);
  }

  function stop() {
    const mr = recorderRef.current;
    if (mr && mr.state !== "inactive") mr.stop();
    setRecording(false);
  }

  return (
    <div className="card">
      <div className="cardTitle">Camera</div>

      <div className="videoBox">
        <video ref={videoRef} autoPlay playsInline muted />
      </div>

      <div className="row">
        <button className="btn" onClick={start} disabled={recording}>Start</button>
        <button className="btn" onClick={stop} disabled={!recording}>Stop</button>
      </div>

      {previewUrl && (
        <>
          <div className="muted">Recorded preview:</div>
          <div className="videoBox">
            <video src={previewUrl} controls />
          </div>
        </>
      )}
    </div>
  );
}
