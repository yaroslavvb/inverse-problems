const SVG_NS = "http://www.w3.org/2000/svg";

// Four-corner regions are calibrated in the original 1536 x 2048 image space.
// Keeping the exact quadrilateral for each fragment avoids perspective drift.
const words = [
  { id: "l1-dash", line: 1, text: "—", ariaLabel: "dash", points: [[790.6, 1445.9], [797.2, 1446.3], [797.2, 1450.4], [790.6, 1450]], confidence: "high" },
  { id: "l1-if", line: 1, text: "if", points: [[798.8, 1442.9], [805.8, 1443.3], [805.8, 1453.4], [798.8, 1453]], confidence: "high" },
  { id: "l1-i", line: 1, text: "I", points: [[807.2, 1442.9], [812.1, 1443.2], [812.1, 1453.3], [807.2, 1453]], confidence: "high" },
  { id: "l1-can", line: 1, text: "can", points: [[812.7, 1446], [823.5, 1446.7], [823.5, 1454.5], [812.7, 1453.8]], confidence: "high" },
  { id: "l1-model", line: 1, text: "model", points: [[825, 1447.1], [843.8, 1448.2], [843.8, 1456.3], [825, 1455.2]], confidence: "high" },
  { id: "l1-user", line: 1, text: "user", points: [[848, 1450.1], [860, 1450.8], [860, 1458.1], [848, 1457.4]], confidence: "high" },
  { id: "l1-engagement", line: 1, text: "engagement", points: [[863.2, 1451.5], [890, 1453.1], [890, 1460.5], [863.2, 1458.9]], confidence: "high" },
  { id: "l1-as", line: 1, text: "as", points: [[897, 1452.5], [908, 1453.2], [908, 1460.8], [897, 1460.1]], confidence: "medium" },

  { id: "l2-more", line: 2, text: "more", points: [[797.5, 1453.8], [813.5, 1454.7], [813.5, 1461.2], [797.5, 1460.2]], confidence: "high" },
  { id: "l2-like", line: 2, text: "like", points: [[816, 1454.6], [823.3, 1455], [823.3, 1462], [816, 1461.6]], confidence: "high" },
  { id: "l2-a", line: 2, text: "a", points: [[826.2, 1457], [830.6, 1457.3], [830.6, 1462], [826.2, 1461.7]], confidence: "high" },
  { id: "l2-person", line: 2, text: "person", points: [[835.5, 1456.8], [851.2, 1457.7], [851.2, 1466.3], [835.5, 1465.5]], confidence: "high" },
  { id: "l2-actually", line: 2, text: "actually", points: [[859.5, 1458.3], [882.8, 1459.7], [882.8, 1467.5], [859.5, 1466]], confidence: "high" },
  { id: "l2-in", line: 2, text: "in", points: [[886.2, 1460], [891.7, 1460.3], [891.7, 1467.5], [886.2, 1467.2]], confidence: "high" },
  { id: "l2-your", line: 2, text: "your", points: [[897, 1460], [908.5, 1460.7], [908.5, 1469.7], [897, 1469]], confidence: "medium" },

  { id: "l3-life", line: 3, text: "life", points: [[797, 1459.5], [808.5, 1460.2], [808.5, 1468.9], [797, 1468.2]], confidence: "high" },
  { id: "l3-than", line: 3, text: "than", points: [[811, 1462], [821.7, 1462.6], [821.7, 1471.2], [811, 1470.6]], confidence: "high" },
  { id: "l3-a", line: 3, text: "a", points: [[825.4, 1465.3], [830, 1465.6], [830, 1472.1], [825.4, 1471.8]], confidence: "high" },
  { id: "l3-few", line: 3, text: "few", points: [[835.2, 1463.8], [847.2, 1464.5], [847.2, 1472.3], [835.2, 1471.6]], confidence: "high" },
  { id: "l3-texts", line: 3, text: "texts", points: [[851.5, 1466], [870.2, 1467.2], [870.2, 1474.2], [851.5, 1473]], confidence: "medium" },

  { id: "l4-dash", line: 4, text: "—", ariaLabel: "dash", points: [[783.5, 1473.5], [788.2, 1473.8], [788.2, 1476.8], [783.5, 1476.5]], confidence: "high" },
  { id: "l4-build", line: 4, text: "build", points: [[792, 1473], [807, 1473.9], [807, 1481.5], [792, 1480.6]], confidence: "high" },
  { id: "l4-the", line: 4, text: "the", points: [[812.5, 1474.5], [823.5, 1475.1], [823.5, 1482.8], [812.5, 1482.2]], confidence: "high" },
  { id: "l4-flow", line: 4, text: "flow", points: [[824.2, 1475], [834.7, 1475.7], [834.7, 1483.7], [824.2, 1483]], confidence: "high" },
  { id: "l4-for", line: 4, text: "for", points: [[840, 1477.2], [847, 1477.6], [847, 1484.6], [840, 1484.2]], confidence: "high" },
  { id: "l4-every", line: 4, text: "every", points: [[851.7, 1478.8], [871.2, 1480], [871.2, 1488.4], [851.7, 1487.2]], confidence: "high" },
  { id: "l4-call-msg", line: 4, text: "call/msg", points: [[875, 1480.5], [899.2, 1481.9], [899.2, 1490.8], [875, 1489.4]], confidence: "medium" },

  { id: "l5-500", line: 5, text: "500", points: [[794.1, 1479.5], [808.7, 1480.3], [808.7, 1487.5], [794.1, 1486.7]], confidence: "high" },
  { id: "l5-users", line: 5, text: "users", points: [[811.3, 1482.3], [829.7, 1483.4], [829.7, 1491.1], [811.3, 1490]], confidence: "high" },
  { id: "l5-at", line: 5, text: "at", points: [[834, 1485], [843, 1485.5], [843, 1492.7], [834, 1492.2]], confidence: "high" },
  { id: "l5-60", line: 5, text: "$60", points: [[848, 1486.5], [865.2, 1487.5], [865.2, 1494.6], [848, 1493.6]], confidence: "high" },
  { id: "l5-a", line: 5, text: "a", points: [[867.4, 1490], [872.5, 1490.3], [872.5, 1495.3], [867.4, 1495]], confidence: "high" },
  { id: "l5-yr", line: 5, text: "yr", points: [[875.1, 1489], [886.5, 1489.7], [886.5, 1498.5], [875.1, 1497.8]], confidence: "high" },
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
  if (activeId) positionHoverCard(highlightElements.get(activeId));
});

mappedCount.textContent = String(words.length);
renderZones();
renderTranscript();
