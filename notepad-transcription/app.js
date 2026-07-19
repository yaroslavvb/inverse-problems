const SVG_NS = "http://www.w3.org/2000/svg";
const slope = 0.055;

const words = [
  { id: "l1-if", line: 1, text: "if", x1: 790, x2: 797, y: 1445, confidence: "high" },
  { id: "l1-i", line: 1, text: "I", x1: 799, x2: 803, y: 1445.5, confidence: "high" },
  { id: "l1-can", line: 1, text: "can", x1: 805, x2: 815, y: 1445.8, confidence: "high" },
  { id: "l1-model", line: 1, text: "model", x1: 818, x2: 834, y: 1446.5, confidence: "high" },
  { id: "l1-user", line: 1, text: "user", x1: 837, x2: 851, y: 1447.5, confidence: "high" },
  { id: "l1-engagement", line: 1, text: "engagement", x1: 854, x2: 886, y: 1448.3, confidence: "high" },
  { id: "l1-as", line: 1, text: "as", x1: 889, x2: 898, y: 1450.1, confidence: "medium" },

  { id: "l2-more", line: 2, text: "more", x1: 789, x2: 803, y: 1458, confidence: "high" },
  { id: "l2-like", line: 2, text: "like", x1: 806, x2: 817, y: 1458.9, confidence: "high" },
  { id: "l2-a", line: 2, text: "a", x1: 820, x2: 824, y: 1459.6, confidence: "high" },
  { id: "l2-person", line: 2, text: "person", x1: 827, x2: 846, y: 1460, confidence: "high" },
  { id: "l2-actually", line: 2, text: "actually", x1: 849, x2: 874, y: 1461.1, confidence: "high" },
  { id: "l2-in", line: 2, text: "in", x1: 877, x2: 883, y: 1462.6, confidence: "high" },
  { id: "l2-your", line: 2, text: "your", x1: 886, x2: 899, y: 1462.9, confidence: "medium" },

  { id: "l3-life", line: 3, text: "life", x1: 789, x2: 801, y: 1471, confidence: "high" },
  { id: "l3-than", line: 3, text: "than", x1: 804, x2: 818, y: 1471.8, confidence: "high" },
  { id: "l3-a", line: 3, text: "a", x1: 821, x2: 825, y: 1472.7, confidence: "high" },
  { id: "l3-few", line: 3, text: "few", x1: 828, x2: 839, y: 1473, confidence: "high" },
  { id: "l3-texts", line: 3, text: "texts", x1: 842, x2: 859, y: 1473.8, confidence: "medium" },

  { id: "l4-build", line: 4, text: "build", x1: 786, x2: 803, y: 1487, confidence: "high" },
  { id: "l4-the", line: 4, text: "the", x1: 806, x2: 816, y: 1488, confidence: "high" },
  { id: "l4-flow", line: 4, text: "flow", x1: 819, x2: 832, y: 1488.7, confidence: "high" },
  { id: "l4-for", line: 4, text: "for", x1: 835, x2: 845, y: 1489.5, confidence: "high" },
  { id: "l4-every", line: 4, text: "every", x1: 848, x2: 865, y: 1490.2, confidence: "high" },
  { id: "l4-call-msg", line: 4, text: "call/msg", x1: 868, x2: 901, y: 1491.2, confidence: "medium" },

  { id: "l5-500", line: 5, text: "500", x1: 790, x2: 805, y: 1500, confidence: "high" },
  { id: "l5-users", line: 5, text: "users", x1: 808, x2: 827, y: 1501, confidence: "high" },
  { id: "l5-at", line: 5, text: "at", x1: 830, x2: 838, y: 1502, confidence: "high" },
  { id: "l5-60", line: 5, text: "$60", x1: 841, x2: 856, y: 1502.5, confidence: "high" },
  { id: "l5-a", line: 5, text: "a", x1: 859, x2: 864, y: 1503.4, confidence: "high" },
  { id: "l5-yr", line: 5, text: "yr", x1: 867, x2: 876, y: 1503.8, confidence: "high" },
];

const zones = document.querySelector("#wordZones");
const transcript = document.querySelector("#transcript");
const noteFrame = document.querySelector("#noteFrame");
const hoverCard = document.querySelector("#hoverCard");
const hoverLine = document.querySelector("#hoverLine");
const hoverWord = document.querySelector("#hoverWord");
const hoverConfidence = document.querySelector("#hoverConfidence");
const activeReadout = document.querySelector("#activeReadout");
const revealButton = document.querySelector("#revealButton");
const sourceButton = document.querySelector("#sourceButton");
const sourceDialog = document.querySelector("#sourceDialog");
const closeDialog = document.querySelector("#closeDialog");
const mappedCount = document.querySelector("#mappedCount");

const zoneElements = new Map();
const transcriptElements = new Map();
let pinnedId = null;
let activeId = null;

function polygonPoints(word, padding = 0) {
  const height = 10.5;
  const x1 = word.x1 - padding;
  const x2 = word.x2 + padding;
  const y1 = word.y - padding * 0.45;
  const y2 = word.y + (word.x2 - word.x1) * slope - padding * 0.45;
  const bottomPadding = padding * 0.75;
  return `${x1},${y1} ${x2},${y2} ${x2},${y2 + height + bottomPadding} ${x1},${y1 + height + bottomPadding}`;
}

function makeSvgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function renderZones() {
  words.forEach((word) => {
    const group = makeSvgElement("g", {
      class: "word-group",
      tabindex: "0",
      role: "button",
      "aria-label": `${word.text}, line ${word.line}, ${word.confidence} confidence`,
      "data-word-id": word.id,
    });
    const title = makeSvgElement("title");
    title.textContent = `${word.text} · line ${String(word.line).padStart(2, "0")}`;
    const highlight = makeSvgElement("polygon", {
      class: "word-highlight",
      points: polygonPoints(word, 0.15),
      "aria-hidden": "true",
    });
    const hit = makeSvgElement("polygon", {
      class: "word-hit",
      points: polygonPoints(word, 1.25),
      "aria-hidden": "true",
    });
    group.append(title, highlight, hit);
    zones.append(group);
    zoneElements.set(word.id, group);

    group.addEventListener("pointerenter", () => activate(word.id, group));
    group.addEventListener("pointerleave", () => clearTransient(word.id));
    group.addEventListener("focus", () => activate(word.id, group));
    group.addEventListener("blur", () => clearTransient(word.id));
    group.addEventListener("click", (event) => {
      event.preventDefault();
      togglePin(word.id, group);
    });
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        togglePin(word.id, group);
      }
    });
  });
}

function renderTranscript() {
  words.forEach((word) => {
    const line = transcript.querySelector(`[data-line="${word.line}"] p`);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "transcript-word";
    button.dataset.wordId = word.id;
    button.dataset.confidence = word.confidence;
    button.textContent = word.text;
    button.setAttribute(
      "aria-label",
      `${word.text}, line ${word.line}, ${word.confidence} confidence; highlight on photograph`,
    );
    line.append(button, document.createTextNode(" "));
    transcriptElements.set(word.id, button);

    button.addEventListener("pointerenter", () => activate(word.id, zoneElements.get(word.id)));
    button.addEventListener("pointerleave", () => clearTransient(word.id));
    button.addEventListener("focus", () => activate(word.id, zoneElements.get(word.id)));
    button.addEventListener("blur", () => clearTransient(word.id));
    button.addEventListener("click", () => togglePin(word.id, zoneElements.get(word.id)));
  });
}

function activate(id, anchor) {
  const word = words.find((item) => item.id === id);
  if (!word) return;

  if (activeId && activeId !== id) {
    zoneElements.get(activeId)?.classList.remove("is-active");
    transcriptElements.get(activeId)?.classList.remove("is-active");
  }

  activeId = id;
  zoneElements.get(id)?.classList.add("is-active");
  transcriptElements.get(id)?.classList.add("is-active");
  activeReadout.textContent = `Line ${String(word.line).padStart(2, "0")} · ${word.text}`;
  hoverLine.textContent = `Line ${String(word.line).padStart(2, "0")}`;
  hoverWord.textContent = word.text;
  hoverConfidence.textContent = word.confidence === "medium" ? "Likely · softened or obscured" : "Strong visual match";
  hoverCard.hidden = false;
  positionHoverCard(anchor);
}

function positionHoverCard(anchor) {
  if (!anchor) return;
  requestAnimationFrame(() => {
    const frameRect = noteFrame.getBoundingClientRect();
    const anchorRect = anchor.getBoundingClientRect();
    const desiredX = anchorRect.left - frameRect.left + anchorRect.width / 2;
    const desiredY = anchorRect.top - frameRect.top;
    const halfWidth = hoverCard.offsetWidth / 2;
    const x = Math.min(frameRect.width - halfWidth - 12, Math.max(halfWidth + 12, desiredX));
    const y = Math.max(82, desiredY);
    hoverCard.style.left = `${x}px`;
    hoverCard.style.top = `${y}px`;
  });
}

function clearTransient(id) {
  if (activeId !== id) return;
  if (pinnedId && pinnedId !== id) {
    activate(pinnedId, zoneElements.get(pinnedId));
    activeReadout.textContent += " · pinned";
    return;
  }
  if (pinnedId === id) return;
  clearActive();
}

function clearActive() {
  if (activeId) {
    zoneElements.get(activeId)?.classList.remove("is-active");
    transcriptElements.get(activeId)?.classList.remove("is-active");
  }
  activeId = null;
  hoverCard.hidden = true;
  activeReadout.textContent = "Hover a word to begin";
}

function togglePin(id, anchor) {
  if (pinnedId === id) {
    pinnedId = null;
    clearActive();
    return;
  }
  pinnedId = id;
  activate(id, anchor);
  activeReadout.textContent += " · pinned";
}

revealButton.addEventListener("click", () => {
  const revealed = revealButton.getAttribute("aria-pressed") !== "true";
  revealButton.setAttribute("aria-pressed", String(revealed));
  revealButton.textContent = revealed ? "Hide all zones" : "Reveal all zones";
  zones.classList.toggle("is-revealed", revealed);
});

sourceButton.addEventListener("click", () => sourceDialog.showModal());
closeDialog.addEventListener("click", () => sourceDialog.close());
sourceDialog.addEventListener("click", (event) => {
  if (event.target === sourceDialog) sourceDialog.close();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !sourceDialog.open) {
    pinnedId = null;
    clearActive();
  }
});

window.addEventListener("resize", () => {
  if (activeId) positionHoverCard(zoneElements.get(activeId));
});

mappedCount.textContent = String(words.length);
renderZones();
renderTranscript();
