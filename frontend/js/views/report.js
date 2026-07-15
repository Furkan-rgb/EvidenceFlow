import { exportUrl } from "../api.js";
import { createFindingCard } from "../components/finding-card.js";
import { createEvidenceLink } from "../components/evidence-link.js";
import { deriveManualDecisions } from "../decision-data.js";
import { deriveDocumentEvidence } from "../evidence-data.js";

function asArray(value) {
  if (Array.isArray(value)) return value;
  if (value === null || value === undefined) return [];
  return [value];
}

function humanise(value) {
  return String(value || "section").replaceAll("_", " ").replace(/^./, (character) => character.toUpperCase());
}

function narrativeSections(report) {
  if (Array.isArray(report.sections)) return report.sections;
  if (report.sections && typeof report.sections === "object") {
    return Object.entries(report.sections).map(([title, content]) => ({ title: humanise(title), content }));
  }
  return [
    ["Document overview", report.document_overview || report.document_summary],
    ["Review conclusion", report.review_conclusion || report.conclusion],
    ["Recommended actions", report.recommended_actions],
  ].filter(([, content]) => content).map(([title, content]) => ({ title, content }));
}

function createNarrativeSection(section) {
  const panel = document.createElement("section");
  panel.className = "report-section panel";
  const heading = document.createElement("h2");
  heading.textContent = section.title || humanise(section.name);
  panel.append(heading);
  const content = section.content ?? section.summary ?? section.text ?? section.body;
  if (Array.isArray(content)) {
    const list = document.createElement("ul");
    content.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = typeof item === "string" ? item : item.text || item.description || JSON.stringify(item);
      list.append(li);
    });
    panel.append(list);
  } else {
    const paragraph = document.createElement("p");
    paragraph.textContent = content || "No additional narrative was provided.";
    panel.append(paragraph);
  }
  return panel;
}

export function renderReport(reviewId, review, reportPayload) {
  const report = reportPayload?.report || reportPayload;
  const fragment = document.createDocumentFragment();
  const hero = document.createElement("header");
  hero.className = "report-hero";
  const intro = document.createElement("div");
  const eyebrow = document.createElement("p");
  eyebrow.className = "eyebrow";
  eyebrow.textContent = "Review report";
  const title = document.createElement("h1");
  title.textContent = report.company_name
    || review.company_name
    || review.verified_review?.company_name
    || review.summary?.company_name
    || "Onboarding package";
  const summary = document.createElement("p");
  summary.textContent = report.executive_summary || "EvidenceFlow completed the document review.";
  intro.append(eyebrow, title, summary);
  const actions = document.createElement("div");
  actions.className = "report-actions";
  [["JSON", "json"], ["Markdown", "md"]].forEach(([label, format]) => {
    const link = document.createElement("a");
    link.className = "button button-secondary";
    link.href = exportUrl(reviewId, format);
    link.textContent = `Download ${label}`;
    link.setAttribute("download", "");
    actions.append(link);
  });
  hero.append(intro, actions);

  const statusRow = document.createElement("div");
  statusRow.className = "review-toolbar";
  statusRow.style.marginTop = "1rem";
  const status = String(report.status || report.final_status || "completed");
  const pill = document.createElement("span");
  pill.className = `status-pill status-pill-${status.replaceAll("_", "-")}`;
  pill.textContent = humanise(status);
  const id = document.createElement("p");
  id.className = "review-id";
  id.textContent = `Review ${reviewId}`;
  statusRow.append(pill, id);

  const grid = document.createElement("div");
  grid.className = "report-grid";
  const main = document.createElement("div");
  main.className = "report-main";
  narrativeSections(report).forEach((section) => main.append(createNarrativeSection(section)));
  const findings = report.findings || review.findings || review.verified_review?.findings || [];
  const findingsPanel = document.createElement("section");
  findingsPanel.className = "report-section panel";
  const findingHeading = document.createElement("h2");
  findingHeading.textContent = "Verified findings";
  findingsPanel.append(findingHeading);
  const findingList = document.createElement("div");
  findingList.className = "finding-list";
  if (findings.length && typeof findings[0] === "object") {
    findings.forEach((finding) => findingList.append(createFindingCard(reviewId, finding)));
  } else if (findings.length) {
    const list = document.createElement("ul");
    findings.forEach((finding) => { const li = document.createElement("li"); li.textContent = String(finding); list.append(li); });
    findingList.append(list);
  } else {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No reportable findings.";
    findingList.append(empty);
  }
  findingsPanel.append(findingList);
  main.append(findingsPanel);

  const aside = document.createElement("aside");
  aside.className = "report-aside";
  const policyPanel = document.createElement("section");
  policyPanel.className = "report-section panel";
  const policyHeading = document.createElement("h2");
  policyHeading.textContent = "Policy evidence";
  const policyList = document.createElement("ul");
  policyList.className = "reference-list";
  const sectionPolicyIds = asArray(report.sections).flatMap((section) => section.policy_evidence_ids || []);
  const policyEvidence = report.policy_evidence || review.policy_evidence || report.policy_references || sectionPolicyIds;
  asArray(policyEvidence).forEach((evidence) => {
    const li = document.createElement("li");
    const title = document.createElement("span");
    title.className = "reference-title";
    title.textContent = typeof evidence === "string" ? evidence : evidence.citation || evidence.title || evidence.evidence_id;
    li.append(title);
    if (typeof evidence === "object" && (evidence.section_id || evidence.policy_id)) {
      const meta = document.createElement("span");
      meta.className = "reference-meta";
      meta.textContent = [evidence.policy_id, evidence.section_id && `§${evidence.section_id}`].filter(Boolean).join(" ");
      li.append(meta);
    }
    policyList.append(li);
  });
  if (!policyList.children.length) {
    const li = document.createElement("li");
    li.textContent = "No policy references were required.";
    policyList.append(li);
  }
  policyPanel.append(policyHeading, policyList);

  const decisionsPanel = document.createElement("section");
  decisionsPanel.className = "report-section panel";
  const decisionsHeading = document.createElement("h2");
  decisionsHeading.textContent = "Manual decisions";
  const decisionList = document.createElement("ul");
  decisionList.className = "reference-list";
  const decisions = deriveManualDecisions(review, report);
  decisions.forEach((decision) => {
    const li = document.createElement("li");
    const decisionTitle = document.createElement("span");
    decisionTitle.className = "reference-title";
    decisionTitle.textContent = decision.label;
    li.append(decisionTitle);
    const action = document.createElement("span");
    action.className = "reference-meta";
    action.textContent = `${decision.action_label} · ${decision.outcome}`;
    li.append(action);
    decisionList.append(li);
  });
  if (!decisionList.children.length) {
    const li = document.createElement("li");
    li.textContent = "No manual decisions were recorded.";
    decisionList.append(li);
  }
  decisionsPanel.append(decisionsHeading, decisionList);

  const evidencePanel = document.createElement("section");
  evidencePanel.className = "report-section panel";
  const evidenceHeading = document.createElement("h2");
  evidenceHeading.textContent = "Document evidence";
  const evidenceList = document.createElement("ul");
  evidenceList.className = "reference-list";
  const documentEvidence = deriveDocumentEvidence(review, report);
  documentEvidence.forEach((evidence) => {
    const li = document.createElement("li");
    li.append(createEvidenceLink(reviewId, evidence, evidence.label));
    evidenceList.append(li);
  });
  if (!evidenceList.children.length) {
    const li = document.createElement("li");
    li.textContent = "No document evidence references were available.";
    evidenceList.append(li);
  }
  evidencePanel.append(evidenceHeading, evidenceList);
  aside.append(policyPanel, decisionsPanel, evidencePanel);
  grid.append(main, aside);
  fragment.append(hero, statusRow, grid);
  return fragment;
}
