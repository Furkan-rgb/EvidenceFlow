import { appendFileInstances, removeFileInstance } from "../upload-selection.js";
import {
  DECISION_STAGES,
  POLICY_EXPLANATION,
  REVIEWER_SAMPLES,
  REVIEWER_STEPS,
  V1_BOUNDARY,
  sampleDocumentsPath,
} from "../reviewer-guide.js";

const MAX_FILES = 5;
const MAX_FILE_BYTES = 10 * 1024 * 1024;

function readableBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function renderUpload({ onStart }) {
  const fragment = document.createDocumentFragment();
  const header = document.createElement("header");
  header.className = "page-header";
  header.innerHTML = `
    <p class="eyebrow">Document intelligence with human oversight</p>
    <h1>Review an onboarding package with evidence you can trace.</h1>
    <p class="lede">Upload synthetic business PDFs. EvidenceFlow classifies, extracts and cross-checks them, pausing whenever a reviewer must decide.</p>
  `;

  const layout = document.createElement("div");
  layout.className = "upload-layout";
  const panel = document.createElement("section");
  panel.className = "upload-panel panel";
  panel.setAttribute("aria-labelledby", "upload-heading");
  const dropZone = document.createElement("div");
  dropZone.className = "drop-zone";
  dropZone.innerHTML = `
    <div>
      <div class="drop-icon" aria-hidden="true">↑</div>
      <h2 id="upload-heading">Add PDF documents</h2>
      <p>Drop 1–5 files here, or choose them from your computer.</p>
    </div>
  `;
  const pickerLabel = document.createElement("label");
  pickerLabel.className = "button button-secondary";
  pickerLabel.htmlFor = "file-picker";
  pickerLabel.textContent = "Choose PDFs";
  const picker = document.createElement("input");
  picker.id = "file-picker";
  picker.className = "file-input";
  picker.type = "file";
  picker.accept = ".pdf,application/pdf";
  picker.multiple = true;
  pickerLabel.append(picker);
  dropZone.firstElementChild.append(pickerLabel);

  const selectedSection = document.createElement("section");
  selectedSection.className = "selected-files";
  selectedSection.hidden = true;
  const selectedHeading = document.createElement("h2");
  selectedHeading.textContent = "Selected documents";
  const fileList = document.createElement("ul");
  fileList.className = "file-list";
  const actions = document.createElement("div");
  actions.className = "upload-actions";
  const start = document.createElement("button");
  start.className = "button";
  start.type = "button";
  start.textContent = "Start review";
  actions.append(start);
  selectedSection.append(selectedHeading, fileList, actions);
  const error = document.createElement("p");
  error.className = "error-message";
  error.hidden = true;
  error.setAttribute("role", "alert");
  panel.append(dropZone, error, selectedSection);

  const guidance = document.createElement("aside");
  guidance.className = "guidance-panel panel";
  guidance.innerHTML = `
    <h2>Reviewer walkthrough</h2>
    <p>Start with one checked-in synthetic package. In the file picker, select every PDF inside one of these folders:</p>
  `;
  const sampleList = document.createElement("ul");
  sampleList.className = "reviewer-samples";
  REVIEWER_SAMPLES.forEach(({ bundle, label, outcome }) => {
    const item = document.createElement("li");
    const sampleLabel = document.createElement("strong");
    sampleLabel.textContent = label;
    const path = document.createElement("code");
    path.textContent = sampleDocumentsPath(bundle);
    const description = document.createElement("span");
    description.textContent = outcome;
    item.append(sampleLabel, path, description);
    sampleList.append(item);
  });
  const verifyHeading = document.createElement("h3");
  verifyHeading.textContent = "What to verify";
  const stepList = document.createElement("ol");
  stepList.className = "reviewer-steps";
  REVIEWER_STEPS.forEach((step) => {
    const item = document.createElement("li");
    item.textContent = step;
    stepList.append(item);
  });
  const traceHint = document.createElement("p");
  traceHint.className = "trace-hint";
  traceHint.append("For an implementation review, open ");
  const traceLink = document.createElement("a");
  traceLink.href = "http://127.0.0.1:5000";
  traceLink.target = "_blank";
  traceLink.rel = "noreferrer";
  traceLink.textContent = "MLflow traces";
  traceLink.setAttribute("aria-label", "Open local MLflow traces in a new tab");
  traceHint.append(traceLink, " to inspect stages and latency alongside the UI.");
  guidance.append(sampleList, verifyHeading, stepList, traceHint);

  const explanation = document.createElement("details");
  explanation.className = "decision-explainer panel";
  explanation.open = true;
  const explanationSummary = document.createElement("summary");
  explanationSummary.innerHTML = `
    <span>How EvidenceFlow reaches a decision</span>
    <small>Follow the evidence, rules, human checkpoints, and policy support</small>
  `;
  const explanationBody = document.createElement("div");
  explanationBody.className = "decision-explainer-body";
  const flow = document.createElement("ol");
  flow.className = "decision-flow";
  DECISION_STAGES.forEach(({ title, detail }) => {
    const item = document.createElement("li");
    const stageTitle = document.createElement("strong");
    stageTitle.textContent = title;
    const stageDetail = document.createElement("span");
    stageDetail.textContent = detail;
    item.append(stageTitle, stageDetail);
    flow.append(item);
  });
  const policyNote = document.createElement("section");
  policyNote.className = "policy-note";
  policyNote.setAttribute("aria-labelledby", "policy-note-heading");
  const policyHeading = document.createElement("h3");
  policyHeading.id = "policy-note-heading";
  policyHeading.textContent = "Policy evidence is not decision logic";
  const policyText = document.createElement("p");
  policyText.textContent = POLICY_EXPLANATION;
  policyNote.append(policyHeading, policyText);
  const boundary = document.createElement("p");
  boundary.className = "v1-boundary";
  boundary.textContent = V1_BOUNDARY;
  explanationBody.append(flow, policyNote, boundary);
  explanation.append(explanationSummary, explanationBody);

  layout.append(panel, guidance);
  fragment.append(header, layout, explanation);

  let selected = [];
  let nextInstanceNumber = 0;
  function createInstanceId() {
    nextInstanceNumber += 1;
    return `upload-${nextInstanceNumber}`;
  }
  function showError(message) {
    error.textContent = message;
    error.hidden = !message;
  }
  function renderFiles() {
    fileList.replaceChildren();
    selectedSection.hidden = selected.length === 0;
    selected.forEach(({ instanceId, file }) => {
      const row = document.createElement("li");
      row.className = "file-row";
      const badge = document.createElement("span");
      badge.className = "file-type";
      badge.textContent = "PDF";
      const details = document.createElement("span");
      details.className = "file-details";
      const name = document.createElement("span");
      name.className = "file-name";
      name.textContent = file.name;
      const size = document.createElement("span");
      size.className = "file-size";
      size.textContent = readableBytes(file.size);
      details.append(name, size);
      const remove = document.createElement("button");
      remove.className = "button button-quiet";
      remove.type = "button";
      remove.textContent = "Remove";
      remove.setAttribute("aria-label", `Remove ${file.name}`);
      remove.addEventListener("click", () => {
        selected = removeFileInstance(selected, instanceId);
        renderFiles();
      });
      row.append(badge, details, remove);
      fileList.append(row);
    });
  }
  function addFiles(fileCollection) {
    const incoming = Array.from(fileCollection);
    const invalid = incoming.find((file) => !file.name.toLowerCase().endsWith(".pdf") || (file.type && file.type !== "application/pdf"));
    if (invalid) return showError(`${invalid.name} is not a PDF file.`);
    const oversized = incoming.find((file) => file.size > MAX_FILE_BYTES);
    if (oversized) return showError(`${oversized.name} exceeds the 10 MB per-file limit.`);
    const merged = appendFileInstances(selected, incoming, createInstanceId);
    if (merged.length > MAX_FILES) return showError(`Choose no more than ${MAX_FILES} PDFs for one review.`);
    selected = merged;
    showError("");
    renderFiles();
  }
  picker.addEventListener("change", () => { addFiles(picker.files); picker.value = ""; });
  ["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.dataset.dragging = "true";
  }));
  ["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.dataset.dragging = "false";
  }));
  dropZone.addEventListener("drop", (event) => addFiles(event.dataTransfer.files));
  start.addEventListener("click", async () => {
    if (!selected.length) return showError("Choose at least one PDF before starting a review.");
    start.disabled = true;
    start.textContent = "Uploading…";
    showError("");
    try {
      await onStart(selected.map(({ file }) => file));
    } catch (caught) {
      showError(caught instanceof Error ? caught.message : "The review could not be started.");
      start.disabled = false;
      start.textContent = "Start review";
    }
  });
  return fragment;
}
