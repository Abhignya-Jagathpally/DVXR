import ModalityInputGroup from "./ModalityInputGroup.jsx";

export default function MetabolicInputs({ inputs, onChange }) {
  return <ModalityInputGroup groupKey="metabolic" inputs={inputs} onChange={onChange} defaultOpen />;
}
