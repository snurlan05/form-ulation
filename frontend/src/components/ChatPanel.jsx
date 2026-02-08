export default function ChatPanel({ messages }) {
  return (
    <div className="card chatCard">
      <div className="cardTitle">Coach</div>

      <div className="chat">
        {messages.length === 0 ? (
          <div className="muted">Record a set and press Analyze.</div>
        ) : (
          messages.map((m, i) => (
            <div key={i} className={`bubble ${m.role === "user" ? "bubbleUser" : "bubbleCoach"}`}>
              {m.text}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
