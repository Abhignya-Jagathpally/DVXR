import { useRef } from "react";
import { brand } from "../content.js";
import { useScrollChrome } from "../lib/useReveal.js";

export default function Nav() {
  const navRef = useRef(null);
  const progRef = useRef(null);
  useScrollChrome(navRef, progRef);
  return (
    <>
      <div className="progress" ref={progRef} />
      <nav id="nav" ref={navRef}>
        <a className="brand" href="#hero"><b>{brand.name}</b></a>
        <div className="links">
          {brand.nav.map((n) => (
            <a key={n.href} href={n.href} data-sec>{n.label}</a>
          ))}
        </div>
        <a className="cta" href="#experience">Launch Prototype</a>
      </nav>
    </>
  );
}
