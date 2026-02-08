export default function VideoLinks({ links }) {
  if (!links || links.length === 0) return null;

  return (
    <div className="card">
      <div className="cardTitle">Recommended videos</div>

      <div className="links">
        {links.map((l, i) => (
          <a key={i} className="linkCard" href={l.url} target="_blank" rel="noreferrer">
            <div className="linkTitle">{l.title}</div>
            {l.tag ? <div className="muted">{l.tag}</div> : null}
          </a>
        ))}
      </div>
    </div>
  );
}
