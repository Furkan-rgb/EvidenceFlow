import { createEvidenceLink } from "./evidence-link.js";

const TITLES = {
  missing_document: "Required document missing",
  missing_required_field: "Required information missing",
  field_conflict: "Submitted values conflict",
};

function humanise(value) {
  return String(value || "finding").replaceAll("_", " ").replace(/^./, (character) => character.toUpperCase());
}

function findingDescription(finding) {
  if (finding.message || finding.description) return finding.message || finding.description;
  const type = finding.type || finding.finding_type;
  if (type === "missing_document") return `${humanise(finding.document_type)} was not found in the package.`;
  if (type === "missing_required_field") {
    return `${humanise(finding.field || finding.field_name)} is missing from ${humanise(finding.document_type || "a required document")}.`;
  }
  if (type === "field_conflict") return `${humanise(finding.field || finding.field_name)} differs between submitted documents.`;
  return "EvidenceFlow identified an item that needs attention.";
}

export function createFindingCard(reviewId, finding) {
  const card = document.createElement("article");
  const severity = String(finding.severity || "medium").toLowerCase();
  card.className = "finding-card panel";
  card.dataset.severity = severity;

  const header = document.createElement("div");
  header.className = "finding-card-header";
  const heading = document.createElement("h3");
  const type = finding.type || finding.finding_type;
  heading.textContent = finding.title || TITLES[type] || humanise(type);
  const pill = document.createElement("span");
  pill.className = `severity-pill severity-pill-${severity}`;
  pill.textContent = severity;
  header.append(heading, pill);

  const description = document.createElement("p");
  description.textContent = findingDescription(finding);
  card.append(header, description);

  const evidence = finding.evidence || finding.evidence_references || [];
  if (Array.isArray(evidence) && evidence.length) {
    const links = document.createElement("div");
    links.className = "candidate-list";
    evidence.forEach((item) => {
      const row = document.createElement("div");
      row.className = "candidate";
      const value = document.createElement("div");
      value.className = "candidate-value";
      value.textContent = item.value ?? item.extracted_value ?? "Referenced evidence";
      row.append(value, createEvidenceLink(reviewId, item));
      links.append(row);
    });
    card.append(links);
  }
  return card;
}
