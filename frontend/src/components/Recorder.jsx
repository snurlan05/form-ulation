import { useEffect, useRef, useState } from "react";

export default function Recorder({ onRecorded }) {
  const videoRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);

  const [stream, setStream] = useState(null);
  const [recording, setRecording] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [cameraError, setCameraError] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        setStream(s);
        if (videoRef.current) videoRef.current.srcObject = s;
      } catch {
        // Previously this rejection was unhandled and the camera area
        // just stayed blank. Show the reason instead.
        setCameraError(
          "Could not access the camera. Allow camera access and use a secure (https:// or localhost) page, then reload."
        );
      }
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

      {cameraError ? (
        <div className="muted">{cameraError}</div>
      ) : (
        <div className="videoBox">
          <video ref={videoRef} autoPlay playsInline muted />
        </div>
      )}

      <div className="row">
        <button className="btn" onClick={start} disabled={recording || !stream}>Start</button>
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
