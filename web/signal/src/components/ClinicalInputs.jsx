import ModalityInputGroup from "./ModalityInputGroup.jsx";

export default function ClinicalInputs({ inputs, onChange }) {
  return <ModalityInputGroup groupKey="clinical" inputs={inputs} onChange={onChange} />;
}
