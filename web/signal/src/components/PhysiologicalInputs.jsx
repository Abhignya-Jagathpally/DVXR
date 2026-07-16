import ModalityInputGroup from "./ModalityInputGroup.jsx";

export default function PhysiologicalInputs({ inputs, onChange }) {
  return <ModalityInputGroup groupKey="physiological" inputs={inputs} onChange={onChange} defaultOpen />;
}
