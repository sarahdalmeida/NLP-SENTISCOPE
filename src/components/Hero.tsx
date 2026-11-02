const links = [
  ["Dashboard", "dashboard"],
  ["Sentiment Analysis", "analyser"],
  ["Model Comparison", "models"],
  ["Dataset", "dataset"],
  ["Methodology", "methodology"],
];

export function Hero({
  activeSection,
  onNavigate,
}: {
  activeSection: string;
  onNavigate: (section: string) => void;
}) {
  return (
    <header className="site-header" id="home">
      <a className="brand-lockup" href="#dashboard" aria-label="SentiScope dashboard">
        <span className="brand-symbol" aria-hidden="true">S</span>
        <span className="brand-wordmark">SentiScope</span>
      </a>
      <nav className="site-nav" aria-label="Main navigation">
        {links.map(([label, id]) => (
          <a
            key={id}
            href={`#${id}`}
            className={activeSection === id ? "is-active" : ""}
            aria-current={activeSection === id ? "location" : undefined}
            onClick={() => onNavigate(id)}
          >
            {label}
          </a>
        ))}
      </nav>
    </header>
  );
}
