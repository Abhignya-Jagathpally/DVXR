import { useReveal } from "./lib/useReveal.js";
import { worlds } from "./content.js";
import Nav from "./components/Nav.jsx";
import HeroSection from "./components/HeroSection.jsx";
import ProblemSection from "./components/ProblemSection.jsx";
import SignalWorld from "./components/SignalWorld.jsx";
import IntelligencePipeline from "./components/IntelligencePipeline.jsx";
import InteractiveExperience from "./components/InteractiveExperience.jsx";
import FeatureGrid from "./components/FeatureGrid.jsx";
import UseCases from "./components/UseCases.jsx";
import EvidenceSection from "./components/EvidenceSection.jsx";
import ResearchRoadmap from "./components/ResearchRoadmap.jsx";
import ResearcherProfile from "./components/ResearcherProfile.jsx";
import ResearchDisclaimer from "./components/ResearchDisclaimer.jsx";

export default function App() {
  useReveal();
  return (
    <>
      <Nav />
      <HeroSection />
      <ProblemSection />
      <section id="worlds">
        <div className="wrap pad-y" style={{ paddingBottom: "40px" }}>
          <div className="kicker reveal"><span className="eyebrow">One person · five signal worlds</span></div>
          <h2 className="display chapter-h reveal">Every signal is a<br />different language.</h2>
        </div>
        {worlds.map((w) => <SignalWorld key={w.idx} world={w} />)}
      </section>
      <IntelligencePipeline />
      <InteractiveExperience />
      <FeatureGrid />
      <UseCases />
      <EvidenceSection />
      <ResearchRoadmap />
      <ResearcherProfile />
      <ResearchDisclaimer />
    </>
  );
}
