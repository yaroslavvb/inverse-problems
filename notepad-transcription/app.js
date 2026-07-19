const SVG_NS = "http://www.w3.org/2000/svg";

// Four-corner regions are calibrated in the original 1536 x 2048 image space.
// Keeping the exact quadrilateral for each fragment avoids perspective drift.
const words = [
  { id: "l1-dash", line: 1, text: "—", ariaLabel: "dash", points: [[790.6, 1445.9], [797.2, 1446.3], [797.2, 1450.4], [790.6, 1450]], confidence: "high" },
  { id: "l1-if", line: 1, text: "if", points: [[798.8, 1442.9], [805.8, 1443.3], [805.8, 1453.4], [798.8, 1453]], confidence: "high" },
  { id: "l1-some", line: 1, text: "some", points: [[807.2, 1442.9], [823.5, 1446.7], [823.5, 1454.5], [807.2, 1453]], confidence: "high" },
  { id: "l1-variable", line: 1, text: "variable", points: [[825, 1447.1], [843.8, 1448.2], [843.8, 1456.1], [825, 1455.1]], confidence: "high" },
  { id: "l1-gets", line: 1, text: "gets", points: [[848, 1450.1], [860, 1450.8], [860, 1457.8], [848, 1457]], confidence: "high" },
  { id: "l1-triggered", line: 1, text: "triggered", points: [[863.2, 1451.5], [890, 1453.1], [890, 1460.1], [863.2, 1458.5]], confidence: "high" },
  { id: "l1-and", line: 1, text: "and", points: [[897, 1452.5], [905.2, 1453], [905.2, 1460], [897, 1459.5]], confidence: "medium" },
  { id: "l1-it", line: 1, text: "it", points: [[906, 1453.1], [912, 1453.5], [912, 1460.6], [906, 1460.2]], confidence: "medium" },

  { id: "l2-not", line: 2, text: "not", points: [[797.5, 1453.8], [813.5, 1454.7], [813.5, 1461], [797.5, 1460.2]], confidence: "high" },
  { id: "l2-like", line: 2, text: "like", points: [[816, 1454.6], [823.3, 1455], [823.3, 1461.8], [816, 1461.5]], confidence: "high" },
  { id: "l2-a", line: 2, text: "a", points: [[826.2, 1457], [830.6, 1457.3], [830.6, 1462], [826.2, 1461.7]], confidence: "high" },
  { id: "l2-person", line: 2, text: "person", points: [[835.5, 1456.8], [851.2, 1457.7], [851.2, 1465.7], [835.5, 1464.7]], confidence: "high" },
  { id: "l2-decides", line: 2, text: "decides", points: [[859.5, 1458.3], [882.8, 1459.7], [882.8, 1467.2], [859.5, 1465.8]], confidence: "high" },
  { id: "l2-to", line: 2, text: "to", points: [[886.2, 1460], [891.7, 1460.3], [891.7, 1467.5], [886.2, 1467.2]], confidence: "high" },
  { id: "l2-turn", line: 2, text: "turn", points: [[897, 1461], [908.5, 1461.7], [908.5, 1469.7], [897, 1469]], confidence: "medium" },

  { id: "l3-left", line: 3, text: "left", points: [[797, 1461], [808.5, 1461.6], [808.5, 1468.9], [797, 1468.2]], confidence: "high" },
  { id: "l3-slash", line: 3, text: "/", ariaLabel: "slash", points: [[811, 1462], [812.2, 1462.1], [810.4, 1470.4], [809.2, 1470.3]], confidence: "medium" },
  { id: "l3-right", line: 3, text: "right", points: [[812.8, 1462.2], [821.7, 1462.7], [821.7, 1471.2], [812.8, 1470.7]], confidence: "medium" },
  { id: "l3-it", line: 3, text: "it", points: [[825.4, 1465.3], [830, 1465.6], [830, 1472.1], [825.4, 1471.8]], confidence: "high" },
  { id: "l3-goes", line: 3, text: "goes", points: [[835.2, 1465.2], [847.2, 1465.9], [847.2, 1472.3], [835.2, 1471.6]], confidence: "high" },
  { id: "l3-straight", line: 3, text: "straight", points: [[851.5, 1466.2], [870.2, 1467.4], [870.2, 1474.2], [851.5, 1473]], confidence: "medium" },

  { id: "l4-dash", line: 4, text: "—", ariaLabel: "dash", points: [[783.5, 1473.5], [788.2, 1473.8], [788.2, 1476.8], [783.5, 1476.5]], confidence: "high" },
  { id: "l4-need", line: 4, text: "need", points: [[792, 1473], [807, 1473.9], [807, 1479.6], [792, 1478.8]], confidence: "high" },
  { id: "l4-an", line: 4, text: "an", points: [[812.5, 1474.5], [823.5, 1475.1], [823.5, 1481.1], [812.5, 1479.9]], confidence: "high" },
  { id: "l4-rng", line: 4, text: "RNG", points: [[824.2, 1475], [834.7, 1475.7], [834.7, 1483], [824.2, 1482.4]], confidence: "medium" },
  { id: "l4-for", line: 4, text: "for", points: [[840, 1477.2], [847, 1477.6], [847, 1484.6], [840, 1484.2]], confidence: "high" },
  { id: "l4-turning", line: 4, text: "turning", points: [[851.7, 1478.8], [871.2, 1480], [871.2, 1487.4], [851.7, 1486.2]], confidence: "high" },
  { id: "l4-left", line: 4, text: "left", points: [[875, 1480.5], [885.8, 1481.1], [885.8, 1488.9], [875, 1488.4]], confidence: "medium" },
  { id: "l4-slash", line: 4, text: "/", ariaLabel: "slash", points: [[889.2, 1481.3], [890.8, 1481.4], [888.9, 1490.2], [887.4, 1490.1]], confidence: "medium" },
  { id: "l4-right", line: 4, text: "right", points: [[891.6, 1481.5], [908.5, 1482.5], [908.5, 1491.7], [891.6, 1490.4]], confidence: "medium" },

  { id: "l5-50-a", line: 5, text: "50", points: [[793.2, 1479.4], [803.3, 1480], [803.3, 1487.3], [793.2, 1486.7]], confidence: "high" },
  { id: "l5-50-b", line: 5, text: "50", points: [[804, 1480], [815.5, 1480.7], [815.5, 1488], [804, 1487.3]], confidence: "high" },
  { id: "l5-chance", line: 5, text: "chance", points: [[817, 1482.6], [833, 1483.6], [833, 1491.4], [817, 1490.3]], confidence: "high" },
  { id: "l5-of", line: 5, text: "of", points: [[834, 1485], [843, 1485.5], [843, 1492.7], [834, 1492.2]], confidence: "high" },
  { id: "l5-left", line: 5, text: "left", points: [[848, 1486.5], [865.2, 1487.5], [865.2, 1494.6], [848, 1493.6]], confidence: "high" },
  { id: "l5-or", line: 5, text: "or", points: [[867.4, 1490], [872.5, 1490.3], [872.5, 1495.3], [867.4, 1495]], confidence: "high" },
  { id: "l5-right", line: 5, text: "right", points: [[875.1, 1489.4], [886.5, 1490.1], [886.5, 1498.5], [875.1, 1497.8]], confidence: "medium" },
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
const highlightElements = new Map();
const transcriptElements = new Map();
let pinnedId = null;
let activeId = null;

function polygonPoints(word) {
  return word.points.map(([x, y]) => `${x},${y}`).join(" ");
}

function makeSvgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  return element;
}

function renderZones() {
  words.forEach((word) => {
    const spokenLabel = word.ariaLabel ?? word.text;
    const group = makeSvgElement("g", {
      class: "word-group",
      tabindex: "0",
      role: "button",
      "aria-label": `${spokenLabel}, line ${word.line}, ${word.confidence} confidence`,
      "data-word-id": word.id,
    });
    const title = makeSvgElement("title");
    title.textContent = `${word.text} · line ${String(word.line).padStart(2, "0")}`;
    const highlight = makeSvgElement("polygon", {
      class: "word-highlight",
      points: polygonPoints(word),
      "aria-hidden": "true",
    });
    const hit = makeSvgElement("polygon", {
      class: "word-hit",
      points: polygonPoints(word),
      "aria-hidden": "true",
    });
    group.append(title, highlight, hit);
    zones.append(group);
    zoneElements.set(word.id, group);
    highlightElements.set(word.id, highlight);

    group.addEventListener("pointerenter", () => activate(word.id, highlight));
    group.addEventListener("pointerleave", () => clearTransient(word.id));
    group.addEventListener("focus", () => activate(word.id, highlight));
    group.addEventListener("blur", () => clearTransient(word.id));
    group.addEventListener("click", (event) => {
      event.preventDefault();
      togglePin(word.id, highlight);
    });
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        togglePin(word.id, highlight);
      }
    });
  });
}

function renderTranscript() {
  words.forEach((word) => {
    const spokenLabel = word.ariaLabel ?? word.text;
    const line = transcript.querySelector(`[data-line="${word.line}"] p`);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "transcript-word";
    button.dataset.wordId = word.id;
    button.dataset.confidence = word.confidence;
    button.textContent = word.text;
    button.setAttribute(
      "aria-label",
      `${spokenLabel}, line ${word.line}, ${word.confidence} confidence; highlight on photograph`,
    );
    line.append(button, document.createTextNode(" "));
    transcriptElements.set(word.id, button);

    button.addEventListener("pointerenter", () => activate(word.id, highlightElements.get(word.id)));
    button.addEventListener("pointerleave", () => clearTransient(word.id));
    button.addEventListener("focus", () => activate(word.id, highlightElements.get(word.id)));
    button.addEventListener("blur", () => clearTransient(word.id));
    button.addEventListener("click", () => togglePin(word.id, highlightElements.get(word.id)));
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
    activate(pinnedId, highlightElements.get(pinnedId));
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
  activeReadout.textContent = "Hover a fragment to begin";
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
  if (activeId) positionHoverCard(highlightElements.get(activeId));
});

mappedCount.textContent = String(words.length);
renderZones();
renderTranscript();
