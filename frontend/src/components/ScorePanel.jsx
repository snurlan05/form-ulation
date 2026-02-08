export default function ScorePanel({ score, breakdown }) {
  return (
    <div className="card">
      <div className="cardTitle">Score</div>

      <div className="scoreBig">{score ?? "--"}</div>

      {breakdown && (
        <div className="breakdown">
          {Object.entries(breakdown).map(([k, v]) => (
            <div key={k} className="breakdownRow">
              <div className="muted">{k}</div>
              <div>{v}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
