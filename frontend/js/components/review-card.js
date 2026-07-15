import { createEvidenceExcerpt, createEvidenceLink } from "./evidence-link.js";

const DOCUMENT_TYPES = [
  "application_form",
  "company_extract",
  "financial_statement",
  "supporting_correspondence",
  "unknown",
];

function humanise(value) {
  return String(value || "item").replaceAll("_", " ").replace(/^./, (character) => character.toUpperCase());
}

function itemId(item) {
  return item.review_item_id || item.item_id || item.id;
}

function itemType(item) {
  return item.type || item.item_type || item.review_type;
}

function addChoice(list, groupName, { action, value, selectedFieldId, title, description }) {
  const label = document.createElement("label");
  label.className = "choice";
  const input = document.createElement("input");
  input.type = "radio";
  input.name = groupName;
  input.value = action;
  if (value !== undefined) input.dataset.decisionValue = JSON.stringify(value);
  if (selectedFieldId !== undefined) input.dataset.selectedFieldId = selectedFieldId;
  const copy = document.createElement("span");
  const strong = document.createElement("strong");
  strong.textContent = title;
  copy.append(strong);
  if (description) {
    const small = document.createElement("small");
    small.textContent = description;
    copy.append(small);
  }
  label.append(input, copy);
  list.append(label);
  return input;
}

function addCorrectionControl(container, item, inputId) {
  const wrapper = document.createElement("div");
  wrapper.className = "nested-control";
  const label = document.createElement("label");
  label.className = "form-label";
  label.htmlFor = inputId;
  label.textContent = "Corrected value";
  const expectedType = item.expected_value_type || item.value_type || "string";
  let input;
  if (expectedType === "boolean") {
    input = document.createElement("select");
    input.className = "select-input";
    [["", "Select a value"], ["true", "True"], ["false", "False"]].forEach(([value, text]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = text;
      input.append(option);
    });
  } else {
    input = document.createElement("input");
    input.className = "text-input";
    input.type = ["integer", "number", "decimal"].includes(expectedType) ? "number" : expectedType === "date" ? "date" : "text";
    if (expectedType === "integer") input.step = "1";
    if (["number", "decimal"].includes(expectedType)) input.step = "any";
    input.autocomplete = "off";
  }
  input.id = inputId;
  input.dataset.correctionInput = "true";
  input.dataset.expectedType = expectedType;
  wrapper.append(label, input);
  container.append(wrapper);
  return input;
}

function parseCorrection(input) {
  const value = input.value.trim();
  if (!value) return { ok: false, message: "Enter a corrected value." };
  const expectedType = input.dataset.expectedType;
  if (expectedType === "integer") {
    const parsed = Number(value);
    if (!Number.isSafeInteger(parsed)) return { ok: false, message: "Enter a whole number." };
    return { ok: true, value: parsed };
  }
  if (["number", "decimal"].includes(expectedType)) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return { ok: false, message: "Enter a valid number." };
    return { ok: true, value: parsed };
  }
  if (expectedType === "boolean") return { ok: true, value: value === "true" };
  return { ok: true, value };
}

function renderCandidates(card, reviewId, candidates) {
  if (!Array.isArray(candidates) || !candidates.length) return;
  const list = document.createElement("div");
  list.className = "candidate-list";
  candidates.forEach((candidate) => {
    const row = document.createElement("div");
    row.className = "candidate";
    const source = document.createElement("div");
    source.className = "candidate-source";
    source.textContent = humanise(candidate.document_type || candidate.source_label || "Submitted document");
    const value = document.createElement("div");
    value.className = "candidate-value";
    value.textContent = candidate.value ?? candidate.extracted_value ?? "No value";
    row.append(source, value, createEvidenceLink(reviewId, candidate));
    list.append(row);
  });
  card.append(list);
}

export function createReviewCard(reviewId, item, index) {
  const id = itemId(item);
  const type = itemType(item);
  const card = document.createElement("article");
  card.className = "review-card panel";
  const header = document.createElement("div");
  header.className = "review-card-header";
  const headingGroup = document.createElement("div");
  const heading = document.createElement("h3");
  const fieldName = item.field_name || item.field?.name;
  heading.textContent = type === "low_confidence_classification"
    ? "Document classification"
    : type === "field_conflict"
      ? `${humanise(fieldName)} conflict`
      : humanise(fieldName || "Extracted field");
  const context = document.createElement("p");
  context.textContent = type === "field_conflict" ? "Choose the verified value or leave the conflict unresolved." : "EvidenceFlow needs a reviewer decision before continuing.";
  headingGroup.append(heading, context);
  header.append(headingGroup);
  const confidenceValue = item.confidence ?? item.classification_confidence;
  if (confidenceValue !== undefined && confidenceValue !== null) {
    const confidence = document.createElement("span");
    confidence.className = "confidence";
    confidence.textContent = `${Math.round(Number(confidenceValue) * 100)}% confidence`;
    header.append(confidence);
  }
  card.append(header);

  const extractedValue = item.extracted_value ?? item.proposed_document_type ?? item.value;
  if (extractedValue !== undefined) {
    const value = document.createElement("div");
    value.className = "value-block";
    const label = document.createElement("span");
    label.className = "value-label";
    label.textContent = type === "low_confidence_classification" ? "Proposed document type" : "Extracted value";
    const content = document.createElement("span");
    content.className = "value-content";
    content.textContent = extractedValue === null ? "No value extracted" : String(extractedValue);
    value.append(label, content);
    card.append(value);
  }
  if (item.reasoning_summary) {
    const reasoning = document.createElement("p");
    reasoning.className = "reference-meta";
    reasoning.textContent = item.reasoning_summary;
    card.append(reasoning);
  }

  const evidence = item.evidence || item.source_evidence;
  const firstEvidence = Array.isArray(evidence) ? evidence[0] : evidence;
  if (firstEvidence) {
    const excerpt = createEvidenceExcerpt(firstEvidence);
    if (excerpt) card.append(excerpt);
    card.append(createEvidenceLink(reviewId, firstEvidence));
  } else if (type === "low_confidence_classification" && item.document_id) {
    card.append(createEvidenceLink(reviewId, { document_id: item.document_id }, "Open document"));
  }
  renderCandidates(card, reviewId, item.candidates || item.values || item.options);

  const fieldset = document.createElement("fieldset");
  fieldset.className = "decision-fieldset";
  const legend = document.createElement("legend");
  legend.textContent = "Reviewer decision";
  const choices = document.createElement("div");
  choices.className = "choice-list";
  const groupName = `decision-${index}`;
  let correctionInput = null;
  let classificationSelect = null;

  if (type === "low_confidence_classification") {
    addChoice(choices, groupName, {
      action: "approve",
      title: `Approve ${humanise(item.proposed_document_type)}`,
      description: "Keep the proposed classification.",
    });
    const correctRadio = addChoice(choices, groupName, {
      action: "correct",
      title: "Correct document type",
      description: "Choose the document type that best describes this PDF.",
    });
    const wrapper = document.createElement("div");
    wrapper.className = "nested-control";
    const label = document.createElement("label");
    label.className = "form-label";
    label.htmlFor = `classification-${index}`;
    label.textContent = "Correct document type";
    classificationSelect = document.createElement("select");
    classificationSelect.id = `classification-${index}`;
    classificationSelect.className = "select-input";
    const prompt = document.createElement("option");
    prompt.value = "";
    prompt.textContent = "Select a document type";
    classificationSelect.append(prompt);
    DOCUMENT_TYPES.forEach((documentType) => {
      const option = document.createElement("option");
      option.value = documentType;
      option.textContent = humanise(documentType);
      classificationSelect.append(option);
    });
    classificationSelect.addEventListener("change", () => { if (classificationSelect.value) correctRadio.checked = true; });
    wrapper.append(label, classificationSelect);
    choices.append(wrapper);
  } else if (type === "field_conflict") {
    const candidates = item.candidates || item.values || item.options || [];
    candidates.forEach((candidate, candidateIndex) => {
      const value = candidate.value ?? candidate.extracted_value;
      addChoice(choices, groupName, {
        action: "select_value",
        selectedFieldId: candidate.field_id,
        title: `Use ${String(value)}`,
        description: humanise(candidate.document_type || candidate.source_label || `Candidate ${candidateIndex + 1}`),
      });
    });
    const correctRadio = addChoice(choices, groupName, {
      action: "correct",
      title: "Enter a corrected value",
      description: "Supply a verified value not shown above.",
    });
    correctionInput = addCorrectionControl(choices, item, `correction-${index}`);
    correctionInput.addEventListener("input", () => { if (correctionInput.value) correctRadio.checked = true; });
    addChoice(choices, groupName, {
      action: "mark_unresolved",
      title: "Mark unresolved",
      description: "Keep the discrepancy visible in the final review.",
    });
  } else {
    addChoice(choices, groupName, {
      action: "approve",
      title: "Approve extracted value",
      description: "Confirm that the model output matches the evidence.",
    });
    const correctRadio = addChoice(choices, groupName, {
      action: "correct",
      title: "Correct the value",
      description: "Supply the verified value from the source document.",
    });
    correctionInput = addCorrectionControl(choices, item, `correction-${index}`);
    correctionInput.addEventListener("input", () => { if (correctionInput.value) correctRadio.checked = true; });
  }
  const error = document.createElement("p");
  error.className = "field-error";
  error.hidden = true;
  error.setAttribute("role", "alert");
  fieldset.append(legend, choices, error);
  card.append(fieldset);

  function fail(message) {
    error.textContent = message;
    error.hidden = false;
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    return null;
  }

  function readDecision() {
    error.hidden = true;
    const selected = fieldset.querySelector(`input[name="${groupName}"]:checked`);
    if (!selected) return fail("Choose a decision for this review item.");
    const decision = { review_item_id: id, action: selected.value };
    if (selected.value === "select_value") decision.selected_field_id = selected.dataset.selectedFieldId;
    if (selected.value === "correct") {
      if (classificationSelect) {
        if (!classificationSelect.value) return fail("Select the corrected document type.");
        decision.value = classificationSelect.value;
      } else if (correctionInput) {
        const parsed = parseCorrection(correctionInput);
        if (!parsed.ok) return fail(parsed.message);
        decision.value = parsed.value;
      }
    }
    return decision;
  }

  return { element: card, readDecision };
}
