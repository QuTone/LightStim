"use strict";

const byId = (id) => document.getElementById(id);
const data = JSON.parse(byId("patch-data").textContent);
const svgNS = "http://www.w3.org/2000/svg";
const colors = {
  X: "var(--x)", Y: "var(--y)", Z: "var(--z)",
  mixed: "var(--muted, #4a5d63)", I: "var(--muted, #4a5d63)",
};
const ink = "var(--ink, #182f35)";
const card = "var(--card, #fff)";
const muted = colors.I;
const categories = {stabilizer: "Stabilizers", logical: "Logicals", gauge: "Gauges"};
const layerDefinitions = {
  all: {label: "All"},
  x: {label: "X Stabilizers", category: "stabilizer", basis: "X"},
  z: {label: "Z Stabilizers", category: "stabilizer", basis: "Z"},
  lx: {label: "Logical X", category: "logical", basis: "X"},
  lz: {label: "Logical Z", category: "logical", basis: "Z"},
};
const patches = data.patches ?? [];
const isSystem = Array.isArray(data.patches);
const stabilizers = data.operators.filter((op) => op.category === "stabilizer");
const checks = stabilizers.filter((op) => op.active !== false);
const checksBySyndrome = new Map();
for (const op of checks) {
  if (op.syndrome_qubit == null) continue;
  if (!checksBySyndrome.has(op.syndrome_qubit)) checksBySyndrome.set(op.syndrome_qubit, []);
  checksBySyndrome.get(op.syndrome_qubit).push(op);
}
const logicals = data.operators.filter((op) => op.category === "logical");
const firstCategory = data.operators[0]?.category ?? "stabilizer";
const enabledOverlays = new Set(data.overlays.map((_, i) => i));
const opSupport = new Map(data.operators.map((op) => [op, new Map(op.support)]));
const plot = layout(data.qubits);
let layer = "all";
let category = firstCategory;
let selected = null;
let inspecting = false;
let inspectedQubit = null;
let highlightedPatch = null;

function element(tag, attrs = {}, text = null, namespace = null) {
  const node = namespace ? document.createElementNS(namespace, tag) : document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== null) node.textContent = text;
  return node;
}
const svg = (tag, attrs, text) => element(tag, attrs, text, svgNS);

function roleKind(role) {
  if (role === "data") return "data";
  if (role.startsWith("syndrome") || role === "check") return "check";
  return "other";
}

function layerBasis(op) {
  // An encoded X or Z can have mixed physical factors, e.g. XZZX code.
  return op.category === "logical" && op.declared_type ? op.declared_type : op.basis;
}

function layout(qubits) {
  if (!qubits.length) return {points: new Map(), width: 320, height: 130};
  const rawX = qubits.map((q) => q.x), rawY = qubits.map((q) => q.y);
  const x0 = Math.min(...rawX), y0 = Math.min(...rawY);
  const x1 = Math.max(...rawX), y1 = Math.max(...rawY);
  // Translate before normalizing; large common offsets must not erase small
  // separations. Halving first also handles opposite near-limit coordinates.
  const factor = Number.isFinite(Math.max(x1 - x0, y1 - y0)) ? 1 : 0.5;
  const span = Math.max(x1 * factor - x0 * factor, y1 * factor - y0 * factor) || 1;
  const xs = rawX.map((v) => (v * factor - x0 * factor) / span);
  const ys = rawY.map((v) => (v * factor - y0 * factor) / span);
  // Compute the drawing scale once. Offsets between patches remain in physical units.
  let pitch = Infinity;
  for (let i = 0; i < qubits.length; i++) {
    for (let j = i + 1; j < qubits.length; j++) {
      const d = Math.hypot(xs[j] - xs[i], ys[j] - ys[i]);
      if (d > 0 && d < pitch) pitch = d;
    }
  }
  const scale = Number.isFinite(pitch) ? 40 / Math.max(pitch, 1 / 2000) : 1;
  const extentX = Math.max(...xs) * scale, extentY = Math.max(...ys) * scale;
  const width = Math.max(135, extentX + 54), height = Math.max(105, extentY + 54);
  const left = (width - extentX) / 2, top = (height - extentY) / 2;
  const points = new Map(qubits.map((q, i) => [q.id, {
    x: xs[i] * scale + left, y: ys[i] * scale + top,
  }]));
  return {points, width, height};
}

function hull(points) {
  // The footprint is an envelope of declared support. Collinear sites retain
  // their endpoint segment; a single repeated coordinate remains a point.
  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y)
    .filter((p, i, all) => !i || p.x !== all[i - 1].x || p.y !== all[i - 1].y);
  if (sorted.length < 3) return sorted;
  function edge(order) {
    const path = [];
    for (const p of order) {
      while (path.length >= 2) {
        const a = path[path.length - 2], b = path[path.length - 1];
        if ((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) > 1e-8) break;
        path.pop();
      }
      path.push(p);
    }
    return path.slice(0, -1);
  }
  return edge(sorted).concat(edge(sorted.reverse()));
}

function drawOperator(target, op, state, crossings) {
  const color = colors[op.basis];
  const group = svg("g", {class: "op-shape", "data-operator-id": op.id, "data-state": state,
    "data-category": op.category, "data-basis": op.basis,
    opacity: state === "context" ? 0.28 : 1, fill: color, stroke: color,
    "stroke-width": state === "focus" ? 2.4 : 1.2, "stroke-linejoin": "round"});
  group.append(svg("title", {}, (op.patch ? op.patch + " · " : "") + op.name + " · " + op.pauli_string));
  const points = op.support.map(([id]) => plot.points.get(id));
  if (op.category === "logical" || op.category === "gauge") {
    // Algebra supplies a set of sites, not a geometric traversal. In particular
    // a periodic or disconnected logical must not acquire a line across a gap.
    for (const [id, factor] of op.support) {
      if (crossings?.has(id)) continue;
      const p = plot.points.get(id);
      group.append(svg("circle", {cx: p.x, cy: p.y, r: 8.5, fill: "none", stroke: colors[factor],
        "stroke-width": 2.7, "stroke-dasharray": op.category === "gauge" ? "3 2" : "none"}));
    }
  } else {
    let outline = hull(points);
    const syndrome = plot.points.get(op.syndrome_qubit);
    // A two-site boundary check uses its measurement site as a geometric
    // corner. The signed support still contains exactly the two data sites.
    if (points.length === 2 && outline.length === 2 && syndrome) {
      const triangle = hull([...points, syndrome]);
      if (triangle.length === 3) {
        outline = triangle;
        group.querySelector("title").textContent += " · measurement corner q" + op.syndrome_qubit;
      }
    }
    if (outline.length >= 3) {
      group.append(svg("polygon", {points: outline.map((p) => p.x + "," + p.y).join(" "),
        "fill-opacity": state === "focus" ? 0.5 : 0.32}));
    } else if (outline.length === 2) {
      const [a, b] = outline;
      group.append(svg("line", {x1: a.x, y1: a.y, x2: b.x, y2: b.y,
        "stroke-width": 6, "stroke-opacity": 0.55, "stroke-linecap": "round"}));
    } else if (outline.length === 1) {
      const p = outline[0];
      group.append(svg("circle", {cx: p.x, cy: p.y, r: 9, "fill-opacity": 0.25}));
    }
  }
  if (op.active === false) group.setAttribute("stroke-dasharray", "4 3");
  if (state === "focus") {
    // A focused logical or check can overlay another check on the same site.
    // Let its footprint pass the click to the remaining visible operators.
    group.setAttribute("pointer-events", "none");
  } else {
    group.style.cursor = "pointer";
    group.addEventListener("click", () => focusFromCanvas(op));
  }
  target.append(group);
}

function logicalCrossings(op) {
  // A local X/Z mismatch only witnesses anticommutation for this pair if
  // the total number of local mismatches is odd. Do not mark commuting pairs.
  const intersections = new Map();
  if (op?.category !== "logical") return intersections;
  const factors = opSupport.get(op);
  for (const other of logicals) {
    if (other === op) continue;
    const hits = other.support.filter(([id, factor]) => factors.has(id) && factors.get(id) !== factor);
    if (hits.length % 2 === 0) continue;
    for (const [id, factor] of hits) {
      if (!intersections.has(id)) intersections.set(id, []);
      intersections.get(id).push({factor, name: (other.patch ? other.patch + " · " : "") + other.name});
    }
  }
  return intersections;
}

function showQubit(q) {
  inspectedQubit = q;
  const members = data.overlays.filter((overlay) => overlay.qubits.includes(q.id)).map((o) => o.name);
  if (isSystem) {
    const others = patches.filter((p) => p.id !== q.owner && p.qubits.includes(q.id));
    members.unshift(...others.map((p) => q.roles_by_patch?.[p.id] ? p.id + ": " + q.roles_by_patch[p.id] : p.id));
  }
  const factor = selected && opSupport.get(selected).get(q.id);
  const parts = ["q" + q.id, "(" + q.x + ", " + q.y + ")", q.role];
  if (q.disabled) parts.push("disabled");
  if (q.owner != null) parts.push("owner " + q.owner);
  if (members.length) parts.push(members.join(" / "));
  if (factor) parts.push(factor + " support");
  byId("qubit-info").textContent = parts.join(" · ");
}

function focusFromCanvas(op) {
  category = op.category;
  if (!matchesSearch(op, byId("search").value)) byId("search").value = "";
  selected = op;
  inspecting = true;
  render();
  const list = byId("operator-list");
  const button = list.querySelector(".selected");
  const offset = button.getBoundingClientRect().top - list.getBoundingClientRect().top;
  if (offset < 0 || offset + button.offsetHeight > list.clientHeight) list.scrollTop += offset;
}

function activateQubit(q) {
  showQubit(q);
  if (!selected || roleKind(q.role) !== "check") return;
  const attached = checksBySyndrome.get(q.id);
  if (attached?.length === 1 && attached[0] !== selected) focusFromCanvas(attached[0]);
}

function drawQubit(target, q, factor, crossing, members) {
  const {x, y} = plot.points.get(q.id);
  const kind = roleKind(q.role);
  const group = svg("g", {class: "q", "data-qubit-id": q.id, "data-role": kind,
    opacity: members && !members.has(q.id) ? 0.3 : 1,
    role: "button", tabindex: 0, "aria-label": "Qubit " + q.id + ", " + q.role +
      ", coordinates " + q.x + ", " + q.y + (q.disabled ? ", disabled" : "")});
  const stroke = factor ? colors[factor] : ink;
  const attrs = {class: "glyph", stroke, "stroke-width": factor ? 2.1 : 1.6,
    fill: factor ? colors[factor] : kind === "data" ? ink : card};
  if (kind === "check") {
    group.append(svg("circle", {class: "glyph", cx: x, cy: y, r: 4.9,
      fill: card, stroke: ink, "stroke-width": 1.6}));
  } else if (kind === "data") {
    group.append(svg("circle", {...attrs, cx: x, cy: y, r: 4.9}));
  } else {
    group.append(svg("rect", {...attrs, x: x - 5, y: y - 5, width: 10, height: 10, rx: 2}));
  }
  if (crossing) {
    [8.5, 12].forEach((r, i) => group.append(svg("circle", {class: "logical-crossing",
      cx: x, cy: y, r, fill: "none", stroke: i ? ink : stroke, "stroke-width": 1.3})));
  }
  if (q.disabled) {
    group.append(svg("path", {d: "M " + (x - 5) + " " + (y + 5) + " l 10 -10",
      stroke: muted, "stroke-width": 2.5}));
  }
  const labels = byId("qubit-labels");
  if (!labels || labels.checked) {
    group.append(svg("text", {class: "qubit-label", x, y: y + 17, "text-anchor": "middle",
      "font-size": 9, fill: ink, stroke: card, "stroke-width": 2, "paint-order": "stroke"}, q.id));
  }
  if (factor) {
    group.append(svg("text", {class: "factor-label", x: x + 7, y: y - 7, "font-size": 9,
      "font-weight": 700, fill: ink, stroke: card, "stroke-width": 2, "paint-order": "stroke"}, factor));
  }
  let marker = 0;
  for (const i of enabledOverlays) {
    const overlay = data.overlays[i];
    if (!overlay.qubits.includes(q.id)) continue;
    group.append(svg("rect", {class: "overlay-mark", x: x + 6 + 5 * marker, y: y + 2,
      width: 4, height: 4, rx: 1, fill: overlay.color}));
    marker++;
  }
  const title = "q" + q.id + ": (" + q.x + ", " + q.y + "), " + q.role +
    (q.disabled ? ", disabled" : "") + (factor ? ", " + factor + " support" : "") +
    (crossing ? ", anticommuting overlap with " + crossing.map((c) => c.name).join(", ") : "");
  group.append(svg("title", {}, title));
  group.addEventListener("click", () => activateQubit(q));
  group.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      activateQubit(q);
    }
  });
  target.append(group);
}

function draw() {
  const target = byId("patch-diagram");
  const focusedSite = target.contains(document.activeElement) ? document.activeElement.dataset.qubitId : null;
  target.replaceChildren();
  target.setAttribute("viewBox", "0 0 " + plot.width + " " + plot.height);
  if (!data.qubits.length) {
    target.append(svg("text", {x: 160, y: 65, "text-anchor": "middle", fill: muted, "font-size": 12},
      "No qubit coordinates."));
    return;
  }
  const crossings = logicalCrossings(selected);
  const members = highlightedPatch ? new Set(highlightedPatch.qubits) : null;
  if (selected) {
    for (const op of checks) if (op !== selected) drawOperator(target, op, "context");
    drawOperator(target, selected, "focus", crossings);
  } else if (!inspecting) {
    const filter = layerDefinitions[layer];
    if (filter.category === "logical") {
      for (const check of checks) drawOperator(target, check, "context");
    }
    const operators = filter.category === "logical" ? logicals : checks;
    for (const op of operators) {
      if (!filter.basis || layerBasis(op) === filter.basis) {
        drawOperator(target, op, "overview");
      }
    }
  }
  const support = opSupport.get(selected);
  for (const q of data.qubits) drawQubit(target, q, support?.get(q.id), crossings.get(q.id), members);
  if (inspectedQubit) showQubit(inspectedQubit);
  if (focusedSite != null) target.querySelector('[data-qubit-id="' + focusedSite + '"]').focus({preventScroll: true});
}

function matchesSearch(op, query) {
  const owners = op.patches ?? (op.patch ? [op.patch] : []);
  return [op.name, op.pauli_string, ...owners].join(" ").toLowerCase().includes(query.trim().toLowerCase());
}

function filtered() {
  const query = byId("search").value;
  // While an operator is focused the inspector exposes peers of either basis.
  // The origin layer still governs the overview returned to by the Back button.
  const basis = inspecting ? null : layerDefinitions[layer].basis;
  return data.operators.filter((op) => op.category === category && (!basis || layerBasis(op) === basis) &&
    matchesSearch(op, query));
}

function renderList(visible) {
  for (const tab of byId("tabs").children) {
    tab.setAttribute("aria-pressed", String(tab.dataset.category === category));
  }
  for (const button of byId("layers").children) {
    button.setAttribute("aria-pressed", String(button.dataset.layer === layer));
  }
  for (const button of byId("patch-key").children) {
    button.setAttribute("aria-pressed", String(button.dataset.patch === highlightedPatch?.id));
  }
  const list = byId("operator-list");
  const focusedId = list.contains(document.activeElement) ? document.activeElement.dataset.operatorId : null;
  list.replaceChildren();
  for (const op of visible) {
    const button = element("button", {type: "button", class: "op" + (op === selected ? " selected" : ""),
      "data-operator-id": op.id, "aria-pressed": String(op === selected)});
    const name = element("span", {class: "op-name"});
    const swatch = element("span", {class: "swatch"});
    swatch.style.background = colors[op.basis];
    name.append(swatch, document.createTextNode(op.name));
    if (op.patch) name.append(element("small", {class: "op-patch"}, op.patch));
    button.append(name, element("small", {}, (op.active === false ? "inactive · " : "") + "w " + op.weight));
    button.addEventListener("click", (event) => {
      selected = op;
      inspecting = true;
      render();
      if (event.detail > 0 && window.matchMedia("(max-width: 51.99rem)").matches)
        byId("figure-title").scrollIntoView({block: "start"});
    });
    list.append(button);
    if (focusedId === op.id) button.focus({preventScroll: true});
  }
  if (!visible.length) {
    list.append(element("div", {class: "empty"}, data.operators.length ? "No matching operators." : "No operators declared."));
  }
  if (category === "gauge") list.append(element("div", {class: "group-note"}, "Gauge generators may anticommute."));
}

function details() {
  const back = byId("back-to-layer");
  back.hidden = !inspecting;
  back.textContent = layer === "all" ? "Back to overview" : "Back to " + layerDefinitions[layer].label;
  byId("weight").hidden = !selected;
  byId("pauli").hidden = !selected;
  const overview = !inspecting && data.operators.length > 0;
  byId("operator-detail").hidden = overview;
  byId("category-label").hidden = overview;
  byId("operator-name").hidden = overview;
  if (selected) {
    const op = selected;
    byId("figure-title").textContent = op.name + " · Pauli support";
    byId("category-label").textContent = {stabilizer: "Stabilizer generator", logical: "Logical representative",
      gauge: "Gauge generator"}[op.category];
    byId("operator-name").textContent = op.name;
    byId("weight").textContent = "Weight " + op.weight;
    byId("pauli").textContent = op.pauli_string;
    const parts = op.patch ? [(op.patches ?? [op.patch]).join(" / ")] : [];
    if (op.active !== undefined) parts.push(op.active ? "active" : "inactive");
    if (op.category === "logical" && ["X", "Y", "Z"].includes(op.declared_type)) {
      parts.push("encoded " + op.declared_type);
    }
    parts.push(op.basis === "mixed" ? "Mixed Pauli" : op.basis + "-type", "sign " + (op.sign === 1 ? "+1" : "−1"));
    if (op.syndrome_qubit != null) parts.push("declared syndrome qubit " + op.syndrome_qubit);
    if (op.declared_type && op.declared_type !== op.basis &&
        !(op.category === "logical" && ["X", "Y", "Z"].includes(op.declared_type))) {
      parts.push("record label: " + op.declared_type);
    }
    byId("detail-meta").textContent = parts.join(" · ");
    return;
  }
  byId("pauli").textContent = "—";
  const empty = !data.operators.length;
  if (inspecting || empty) {
    byId("figure-title").textContent = isSystem ? "System geometry" : "Patch geometry";
    byId("category-label").textContent = "Geometry only";
    byId("operator-name").textContent = empty ? "No operators declared" : "No matching operators";
    byId("detail-meta").textContent = empty ? "The " + (isSystem ? "system" : "patch") +
      " has no stabilizer, logical or gauge records." : "No " + categories[category].toLowerCase() +
      " match this filter. Clear or change the search to select an operator.";
    return;
  }
  byId("figure-title").textContent = layer === "all" ? "Code structure" : layerDefinitions[layer].label;
  byId("category-label").textContent = "Overview";
  byId("operator-name").textContent = "Structure overview";
  byId("detail-meta").textContent = "Select an operator to inspect its signed Pauli support.";
}

function render() {
  const visible = filtered();
  if (inspecting && !visible.includes(selected)) selected = visible[0] ?? null;
  renderList(visible);
  details();
  draw();
}

function showLayer(key) {
  layer = key;
  category = layerDefinitions[key].category ?? firstCategory;
  selected = null;
  inspecting = false;
  byId("search").value = "";
  render();
}

function backToLayer() {
  const previousFocus = document.activeElement;
  showLayer(layer);
  // Restore a useful keyboard position when the Back button becomes hidden or
  // a focused operator row disappears from the originating overview's list.
  if (previousFocus === byId("back-to-layer") || (!previousFocus.isConnected && document.activeElement === document.body)) {
    const origin = byId("layers").querySelector('[aria-pressed="true"]');
    origin.focus({preventScroll: true});
  }
}

function renderMetadata() {
  const meta = byId("metadata");
  const row = (label, value) => {
    const div = element("div");
    div.append(element("dt", {}, label), element("dd", {}, value));
    meta.append(div);
  };
  row("Physical qubits", data.qubits.length);
  const roles = new Map();
  for (const q of data.qubits) roles.set(q.role, (roles.get(q.role) ?? 0) + 1);
  const roleOrder = ["data", "syndrome X", "syndrome Z", "syndrome", "other"];
  for (const role of new Set([...roleOrder, ...roles.keys()])) {
    if (roles.has(role)) row(role[0].toUpperCase() + role.slice(1), roles.get(role));
  }
  row("Declared logical qubits", data.num_logicals);
  row("Stabilizer generators", stabilizers.length);
  if (checks.length !== stabilizers.length) row("Active stabilizers", checks.length);
  const weights = new Map();
  for (const op of stabilizers) weights.set(op.weight, (weights.get(op.weight) ?? 0) + 1);
  if (weights.size) row("Stabilizer weights", [...weights.keys()].sort((a, b) => a - b)
    .map((w) => weights.get(w) + " × w" + w).join(" · "));
  const gauges = data.operators.filter((op) => op.category === "gauge");
  if (gauges.length) row("Gauge generators", gauges.length);
  const parameterNames = {distance: "Configured distance", distance_x: "Configured X distance",
    distance_z: "Configured Z distance", boundary_conditions: "Boundary conditions"};
  const declaredParams = (values) => Object.entries(values ?? {}).filter(([key]) => Object.hasOwn(parameterNames, key));
  if (!isSystem) {
    row("Patch", data.patch_class);
    const {distance, distance_x, distance_z} = data.parameters ?? {};
    const tuple = roles.has("data") && stabilizers.length > 0 && !gauges.length && Number.isInteger(distance) && distance > 0 &&
      [distance_x, distance_z].every((d) => d == null || d === distance);
    if (tuple) row("Configured [[n, k, d]]", "[[" + roles.get("data") + ", " + data.num_logicals + ", " + distance + "]]");
    for (const [key, value] of declaredParams(data.parameters)) {
      if (!(tuple && ["distance", "distance_x", "distance_z"].includes(key))) row(parameterNames[key], value);
    }
  } else {
    row("Patches", patches.length);
    const shortNames = {distance: "d", distance_x: "d_x", distance_z: "d_z", boundary_conditions: "boundary"};
    for (const patch of patches) {
      const parts = [patch.patch_class];
      if (patch.kind === "coupler" && !/coupler/i.test(patch.patch_class)) parts.push("coupler");
      for (const [key, value] of declaredParams(patch.parameters)) parts.push(shortNames[key] + " = " + value);
      if (patch.offset) parts.push("(" + patch.offset.join(", ") + ")");
      row(patch.id, parts.join(" · "));
    }
  }
}

for (const [key, definition] of Object.entries(layerDefinitions)) {
  const button = element("button", {type: "button", class: "layer", "data-layer": key}, definition.label);
  if (definition.basis) {
    const operators = definition.category === "logical" ? logicals : checks;
    button.disabled = !operators.some((op) => layerBasis(op) === definition.basis);
    if (button.disabled) button.title = "No " +
      (definition.category === "logical" ? definition.basis + " logical representatives" :
        "active " + definition.basis + " stabilizers") + " in this configuration.";
  }
  button.addEventListener("click", () => showLayer(key));
  byId("layers").append(button);
}
for (const [key, label] of Object.entries(categories)) {
  const count = data.operators.filter((op) => op.category === key).length;
  if (!count) continue;
  const button = element("button", {type: "button", class: "tab", "data-category": key}, label + " " + count);
  button.addEventListener("click", () => {category = key; inspecting = true; render();});
  byId("tabs").append(button);
}
for (const [i, overlay] of data.overlays.entries()) {
  const label = element("label");
  const input = element("input", {type: "checkbox"});
  input.checked = true;
  input.addEventListener("change", () => {
    if (input.checked) enabledOverlays.add(i); else enabledOverlays.delete(i);
    draw();
  });
  const swatch = element("span", {class: "swatch"});
  swatch.style.background = overlay.color;
  label.append(input, swatch, document.createTextNode(overlay.name + " (" + overlay.qubits.length + ")"));
  byId("overlays").append(label);
}
byId("search").addEventListener("input", () => {inspecting = true; render();});
byId("back-to-layer").addEventListener("click", backToLayer);
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || !inspecting || event.defaultPrevented || event.isComposing) return;
  if (event.target instanceof Element && event.target.closest("input,textarea,select,[contenteditable]")) return;
  event.preventDefault();
  backToLayer();
});
byId("qubit-labels")?.addEventListener("change", draw);
for (const [kind, id] of [["check", "legend-aux"], ["other", "legend-other"]]) {
  if (byId(id)) byId(id).hidden = !data.qubits.some((q) => roleKind(q.role) === kind);
}
if (byId("legend-mixed")) byId("legend-mixed").hidden = !data.operators.some((op) => op.basis === "mixed");
if (isSystem) {
  for (const patch of patches) {
    const item = element("button", {type: "button", class: "patch-key-item", "data-patch": patch.id,
      title: "Highlight sites belonging to " + patch.id}, patch.id);
    item.addEventListener("click", () => {
      highlightedPatch = highlightedPatch === patch ? null : patch;
      render();
    });
    byId("patch-key").append(item);
  }
}
renderMetadata();
render();
