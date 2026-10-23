"""Export the algebra and coordinates of a QECPatch as a standalone HTML file.

This viewer is independent of circuit generation. It preserves physical qubit
IDs, coordinates, Pauli factors and signs; support polygons are not gate edges.
It does not infer code distance, stabilizer independence, or an SE schedule.
The Pauli color convention is X red, Z blue, and Y green.

Views are limited to 500 registered qubits, including auxiliary qubits. Only
self-contained patches are supported: each Pauli support must refer to integer
IDs with coordinates in the patch. LogicalCouplerPatch requires surrounding
QECSystem context to resolve its cross-patch support and is not supported here.

Example::

    from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode

    export_patch_html(RotatedSurfaceCode(distance=3), "code.html")

Overlays may overlap. Each has a name, an iterable of existing qubit IDs and an
optional six-digit hex color. They are annotations, not additional operators.
They are supplied explicitly by the caller and are empty by default. The viewer
does not infer regions or read code-specific subclass metadata.

CLI::

    python -m lightstim.frontend.patch_html \\
        --factory lightstim.qec_code.surface_code.rotated:RotatedSurfaceCode \\
        --kwargs '{"distance": 3}' --output code.html
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
import importlib
import json
import math
from pathlib import Path
import re
from typing import Any

import stim

from lightstim.ir.qec_patch import QECPatch


_COLORS = ("#188578", "#a76b20", "#7553ab", "#4775a5")
_MAX_QUBITS = 500


@dataclass(frozen=True)
class PatchVisualization:
    """A standalone HTML snapshot, displayable in a trusted notebook.

    Leave this object as the cell's last expression, or pass it to ``display``.
    Each notebook display uses its own iframe so viewer scripts, element IDs
    and styles remain isolated. No IPython dependency or server is required.
    ``html`` and ``str(view)`` expose the complete standalone document.
    """

    html: str

    def __repr__(self) -> str:
        return "<PatchVisualization: interactive HTML; use .save_html(path) to export>"

    def __str__(self) -> str:
        return self.html

    def save_html(self, output: str | Path) -> Path:
        """Save this snapshot, creating parent folders, and return its path."""
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.html, encoding="utf-8")
        return path

    def _repr_html_(self) -> str:
        # Only scripts are enabled; the embedded document cannot access the
        # notebook's DOM. Attribute escaping keeps the entire document in srcdoc.
        return (
            '<iframe title="LightStim QEC patch" sandbox="allow-scripts" '
            'style="width:100%;height:850px;border:0" '
            f'srcdoc="{escape(self.html, quote=True)}"></iframe>'
        )


def _qubit_id(value: Any) -> int:
    # numpy integers are valid IDs; floats and strings would silently remap IDs.
    import operator

    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"Qubit ID must be an integer, got {value!r}.") from exc
    if isinstance(value, bool) or result < 0:
        raise ValueError(f"Qubit ID must be a nonnegative integer, got {value!r}.")
    return result


def _operator(record: Mapping, category: str, index: int, known: set[int],
              names: Counter) -> dict:
    pauli = record.get("pauli")
    phase = complex(record.get("sign", 1))
    if "phase" in record:
        raise ValueError("Use a signed stim.PauliString or sign=+1/-1; phase metadata is ambiguous.")
    if isinstance(pauli, stim.PauliString):
        phase *= pauli.sign
        support = {q: "_XYZ"[pauli[q]] for q in pauli.pauli_indices()}
    elif isinstance(pauli, Mapping):
        support = {}
        for raw_q, factor in pauli.items():
            q = _qubit_id(raw_q)
            if factor not in ("I", "_", "X", "Y", "Z"):
                raise ValueError(f"Invalid Pauli factor {factor!r} on qubit {q}.")
            if factor in ("X", "Y", "Z"):
                support[q] = factor
    else:
        raise ValueError("Operator 'pauli' must be a Pauli dictionary or stim.PauliString.")
    if phase not in (1, -1):
        raise ValueError("Displayed stabilizers, gauges and logical observables must have real sign +1 or -1.")
    missing = set(support) - known
    if missing:
        raise ValueError(f"Operator references qubits without coordinates: {sorted(missing)}.")
    factors = set(support.values())
    basis = next(iter(factors)) if len(factors) == 1 else ("mixed" if factors else "I")
    declared_type = str(record.get("type") or "")
    names[(category, declared_type or basis)] += 1
    number = names[(category, declared_type or basis)]
    prefix = {"stabilizer": "S", "logical": "L", "gauge": "G"}[category]
    default_name = f"{prefix}_{declared_type or basis}{number}"
    name = str(record.get("name") or record.get("label") or default_name)
    sign = 1 if phase == 1 else -1
    pairs = [[q, support[q]] for q in sorted(support)]
    syndrome = record.get("syn_idx")
    if syndrome is not None:
        syndrome = _qubit_id(syndrome)
        if syndrome not in known:
            raise ValueError(f"Syndrome qubit {syndrome} has no coordinates.")
    return {
        "id": f"{category}-{index}", "name": name, "category": category,
        "basis": basis, "declared_type": declared_type, "sign": sign,
        "support": pairs, "weight": len(pairs), "syndrome_qubit": syndrome,
        "pauli_string": ("+" if sign == 1 else "−") + " ".join(
            f"{factor}{q}" for q, factor in pairs
        ) if pairs else ("+I" if sign == 1 else "−I"),
    }


def patch_view_data(patch: QECPatch, *, title: str | None = None,
                    overlays: Sequence[Mapping] | None = None) -> dict:
    """Return the JSON-compatible viewer snapshot without modifying ``patch``.

    Operator records accept QECPatch Pauli dictionaries (optionally ``sign``
    +/-1) or signed Stim PauliStrings. ``type`` is a label, never a replacement
    for the actual Pauli factors. Gauges retain their separate category.
    Unregistered support and non-Hermitian operator phases are rejected.
    ``name`` and ``label`` are optional display metadata. Unnamed records get
    generated labels such as ``S_X1``, ``S_Z1`` or ``L_X1``. The exporter reads
    only the patch object and supplied overlays, not the patch's source files.
    Self-contained patches with at most 500 registered qubits are supported.
    Coupler patches require QECSystem context and raise ``ValueError``.
    """
    if not isinstance(patch, QECPatch):
        raise TypeError("Expected a QECPatch instance.")
    # Bound work before copying operators or building an HTML document. Count
    # every displayed qubit, including syndrome/flag qubits, not only data.
    if len(patch.qubit_coords) > _MAX_QUBITS:
        raise ValueError(
            f"Patch visualization supports at most {_MAX_QUBITS} registered qubits "
            f"(including auxiliary qubits); got {len(patch.qubit_coords)}."
        )
    from lightstim.ir.coupler import LogicalCouplerPatch

    # Couplers can refer to qubits owned by other patches, even after system
    # registration. Resolving those references needs a system-level viewer;
    # inventing local IDs here would misrepresent the physical support.
    if isinstance(patch, LogicalCouplerPatch):
        raise ValueError(
            "Patch visualization supports self-contained patches only; coupler "
            "patches require surrounding QECSystem geometry to resolve cross-patch support."
        )
    qubits = []
    for raw_q, coord in sorted(patch.qubit_coords.items()):
        q = _qubit_id(raw_q)
        if len(coord) != 2 or not all(math.isfinite(float(v)) for v in coord):
            raise ValueError(f"Qubit {q} requires two finite coordinates.")
        if q in patch.data_indices:
            role = "data"
        elif q in patch.syndrome_indices_x:
            role = "syndrome X"
        elif q in patch.syndrome_indices_z:
            role = "syndrome Z"
        elif q in patch.syndrome_indices:
            role = "syndrome"
        else:
            role = "other"
        qubits.append({"id": q, "x": float(coord[0]), "y": float(coord[1]), "role": role})
    known = {q["id"] for q in qubits}
    names = Counter()
    operators = []
    for category, records in (("stabilizer", patch.stabilizers),
                              ("logical", patch.logical_ops),
                              ("gauge", getattr(patch, "gauges", []))):
        operators.extend(_operator(record, category, i, known, names)
                         for i, record in enumerate(records))
    annotations = []
    for i, overlay in enumerate(overlays or []):
        color = overlay.get("color", _COLORS[i % len(_COLORS)])
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("Overlay color must be a six-digit hex color, e.g. #188578.")
        members = sorted({_qubit_id(q) for q in overlay["qubits"]})
        if set(members) - known:
            raise ValueError(f"Overlay references unknown qubits: {sorted(set(members) - known)}.")
        annotations.append({"name": str(overlay["name"]), "qubits": members, "color": color})
    return {
        "schema": "lightstim.qec-patch-view.v1",
        "title": str(title or type(patch).__name__), "patch_class": type(patch).__name__,
        "num_logicals": int(patch.num_logicals), "qubits": qubits,
        "operators": operators, "overlays": annotations,
    }


def patch_html(patch: QECPatch, *, title: str | None = None,
               overlays: Sequence[Mapping] | None = None) -> str:
    """Render a self-contained document; usable from file:// without a server."""
    data = patch_view_data(patch, title=title, overlays=overlays)
    # JSON is raw text inside a script element, so HTML-escape is not appropriate.
    payload = json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    return _HTML.replace("__TITLE__", escape(data["title"])).replace("__PAYLOAD__", payload)


def export_patch_html(patch: QECPatch, output: str | Path, *, title: str | None = None,
                      overlays: Sequence[Mapping] | None = None) -> Path:
    """Write the standalone viewer and return its path, creating parent folders."""
    return PatchVisualization(patch_html(patch, title=title, overlays=overlays)).save_html(output)


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · LightStim patch viewer</title>
<style>
:root{--ink:#182b3e;--muted:#687b8e;--line:#dbe4eb;--blue:#4269b1;--green:#138176;--red:#c94f54;--paper:#fff;--bg:#f3f6fa}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--bg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}button,input{font:inherit}button{cursor:pointer}header{max-width:1440px;margin:auto;padding:32px 32px 20px}.eyebrow{font-size:11px;font-weight:750;letter-spacing:.15em;color:var(--muted);text-transform:uppercase}h1{font-size:28px;line-height:1.2;margin:9px 0 12px;font-weight:650}.sub{color:var(--muted);margin:0}.stats{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}.stat{border:1px solid var(--line);background:var(--paper);border-radius:6px;padding:5px 11px;font-size:13px}.stat strong{margin-right:5px}main{max-width:1440px;margin:auto;padding:0 32px 30px;display:grid;grid-template-columns:270px minmax(0,1fr);gap:18px}.panel{border:1px solid var(--line);border-radius:12px;background:var(--paper);overflow:hidden}.side{align-self:start;position:sticky;top:18px}.side-head{padding:18px;border-bottom:1px solid var(--line)}h2{font-size:15px;margin:0 0 12px}.tabs{display:flex;flex-wrap:wrap;gap:5px}.tab{border:1px solid var(--line);border-radius:5px;background:#fff;color:var(--muted);font-size:12px;padding:5px 7px}.tab[aria-pressed=true]{color:#254d8d;background:#edf3fc;border-color:#bed0e9}#search{display:block;width:100%;margin-top:12px;border:1px solid var(--line);border-radius:5px;padding:7px 9px;color:var(--ink)}#operator-list{padding:8px;max-height:580px;overflow:auto}.op{width:100%;border:1px solid transparent;background:transparent;border-radius:7px;text-align:left;padding:10px;display:flex;align-items:center;justify-content:space-between;gap:8px;color:var(--ink)}.op:hover{background:#f6f8fb}.op.selected{background:#edf3fc;border-color:#d1deef}.op small{color:var(--muted);font-size:11px}.op-name{display:flex;align-items:center;gap:8px;overflow-wrap:anywhere}.swatch{width:8px;height:8px;border-radius:50%;flex-shrink:0}.group-note{font-size:12px;color:var(--muted);padding:10px 12px}.content{display:grid;gap:16px;align-content:start}.diagram-head{display:flex;justify-content:space-between;gap:10px;align-items:center;padding:16px 20px;border-bottom:1px solid var(--line)}.diagram-head h2{margin:0}.legend{display:flex;gap:13px;flex-wrap:wrap;font-size:12px;color:var(--muted)}.legend span{display:flex;gap:5px;align-items:center}.diagram-wrap{background:linear-gradient(#fff,#fafcfe);padding:14px 16px 0;overflow:auto}#patch-diagram{display:block;width:100%;height:480px;min-width:340px}.overlays{padding:8px 20px 14px;display:flex;flex-wrap:wrap;gap:8px 16px;font-size:12px}.overlays label{display:flex;align-items:center;gap:5px;cursor:pointer}input[type=checkbox]{accent-color:#416faa}.figure-note{font-size:12px;color:var(--muted);padding:12px 20px;margin:0;border-top:1px solid var(--line)}.details{padding:19px 22px}.details-head{display:flex;gap:14px;justify-content:space-between;align-items:start}.details h2{font-size:19px;margin:0 0 8px}.tag{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}#pauli{font:15px/1.8 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere;padding:13px 15px;border:1px solid #e4eaf1;background:#f7f9fc;border-radius:7px;margin:12px 0}#detail-meta{color:var(--muted);font-size:13px}#qubit-info{font:12px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--muted);min-height:20px;margin:10px 0 0}footer{max-width:1440px;margin:auto;padding:0 32px 26px;color:var(--muted);font-size:12px}svg text{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}svg .q{cursor:pointer}svg .q:focus{outline:none}svg .q:focus circle,svg .q:hover circle{stroke-width:3.5}.empty{padding:15px;color:var(--muted);font-size:13px}@media(max-width:800px){header{padding:22px 18px 18px}main{padding:0 18px 22px;grid-template-columns:1fr}.side{position:static}#operator-list{max-height:220px}.diagram-head{align-items:start;flex-direction:column}#patch-diagram{height:420px}footer{padding:0 18px 20px}h1{font-size:24px}}
</style></head><body>
<header><div class="eyebrow">LightStim · QEC patch</div><h1 id="title">__TITLE__</h1><p class="sub">Code geometry, stabilizer generators and logical representatives.</p><div class="stats" id="stats"></div></header>
<main><aside class="panel side"><div class="side-head"><h2>Choose an operator</h2><div class="tabs" id="tabs"></div><input id="search" type="search" placeholder="Name or Pauli support…" aria-label="Filter operators"></div><div id="operator-list"></div></aside>
<section class="content"><div class="panel"><div class="diagram-head"><h2 id="figure-title">Patch geometry</h2><div class="legend"><span><i class="swatch" style="background:var(--red)"></i>X</span><span><i class="swatch" style="background:var(--green)"></i>Y</span><span><i class="swatch" style="background:var(--blue)"></i>Z</span><span>○ data</span><span>□ syndrome / other</span></div></div><div class="diagram-wrap"><svg id="patch-diagram" role="img" aria-label="Qubit geometry and selected Pauli support"></svg></div><div class="overlays" id="overlays"></div><p class="figure-note">Coordinates and qubit IDs come directly from the patch. Shading marks Pauli support; it does not represent gate connections or an execution schedule.</p></div>
<div class="panel details"><div class="details-head"><div><div class="tag" id="category-label"></div><h2 id="operator-name"></h2></div><span class="stat" id="weight"></span></div><div id="pauli"></div><div id="detail-meta"></div><p id="qubit-info">Select a qubit to see its coordinates and region memberships.</p></div></section></main>
<footer>Stabilizers, logical representatives and gauges are displayed as declared. This viewer does not certify independence, commutation, distance or fault tolerance.</footer>
<script id="patch-data" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const DATA=JSON.parse(document.getElementById('patch-data').textContent);
const $=id=>document.getElementById(id), PALETTE=getComputedStyle(document.documentElement);
// Shared convention for legend, list, qubits and support shading: X red, Z blue.
const COLORS={X:PALETTE.getPropertyValue('--red').trim(),Y:PALETTE.getPropertyValue('--green').trim(),Z:PALETTE.getPropertyValue('--blue').trim(),mixed:'#7553ab',I:'#8997a5'};
let selected=DATA.operators[0]||null, category='stabilizer';
if(!DATA.operators.some(o=>o.category===category)) category=selected?selected.category:'stabilizer';
const enabled=new Set(DATA.overlays.map((_,i)=>i));
function el(tag,attrs={},text=null){const n=document.createElement(tag);for(const [k,v] of Object.entries(attrs)) n.setAttribute(k,v);if(text!==null)n.textContent=text;return n;}
function svg(tag,attrs={},text=null){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);if(text!==null)n.textContent=text;return n;}
function stat(value,label){const n=el('span',{class:'stat'});n.append(el('strong',{},String(value)),document.createTextNode(label));$('stats').append(n);}
$('title').textContent=DATA.title;
stat(DATA.qubits.filter(q=>q.role==='data').length,'data qubits');
const auxiliary=DATA.qubits.filter(q=>q.role!=='data').length;if(auxiliary)stat(auxiliary,'auxiliary qubits');
stat(DATA.operators.filter(o=>o.category==='stabilizer').length,'stabilizer generators');stat(DATA.num_logicals,'logical qubits');
if(DATA.operators.some(o=>o.category==='gauge'))stat(DATA.operators.filter(o=>o.category==='gauge').length,'gauge generators');
const labels={stabilizer:'Stabilizers',logical:'Logicals',gauge:'Gauges'};
for(const key of Object.keys(labels)){const count=DATA.operators.filter(o=>o.category===key).length;if(!count)continue;const b=el('button',{class:'tab',type:'button','data-category':key},`${labels[key]} ${count}`);b.addEventListener('click',()=>{category=key;render();});$('tabs').append(b);}
for(const [i,overlay] of DATA.overlays.entries()){const label=el('label'), input=el('input',{type:'checkbox',checked:'checked'});input.addEventListener('change',()=>{if(input.checked)enabled.add(i);else enabled.delete(i);draw();});label.append(input,el('span',{class:'swatch',style:`background:${overlay.color}`}),document.createTextNode(`${overlay.name} (${overlay.qubits.length})`));$('overlays').append(label);}
$('search').addEventListener('input',render);
function filtered(){const query=$('search').value.trim().toLowerCase();return DATA.operators.filter(o=>o.category===category&&(`${o.name} ${o.pauli_string}`).toLowerCase().includes(query));}
function renderList(visible){for(const b of $('tabs').children)b.setAttribute('aria-pressed',String(b.dataset.category===category));$('operator-list').replaceChildren();for(const o of visible){const b=el('button',{type:'button',class:'op'+(o===selected?' selected':''),'data-operator-id':o.id,'aria-pressed':String(o===selected)}),name=el('span',{class:'op-name'});name.append(el('span',{class:'swatch',style:`background:${COLORS[o.basis]}`}),document.createTextNode(o.name));b.append(name,el('small',{},`w ${o.weight}`));b.addEventListener('click',()=>{selected=o;render();});$('operator-list').append(b);}if(!visible.length)$('operator-list').append(el('div',{class:'empty'},DATA.operators.length?'No matching operators.':'No operators declared.'));if(category==='gauge')$('operator-list').append(el('div',{class:'group-note'},'Gauge generators need not commute with each other. They are not being displayed as stabilizers.'));}
function hull(points){if(points.length<3)return points;const p=points.slice().sort((a,b)=>a[0]-b[0]||a[1]-b[1]),cross=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);let lower=[],upper=[];for(const v of p){while(lower.length>=2&&cross(lower.at(-2),lower.at(-1),v)<=0)lower.pop();lower.push(v);}for(const v of p.slice().reverse()){while(upper.length>=2&&cross(upper.at(-2),upper.at(-1),v)<=0)upper.pop();upper.push(v);}lower.pop();upper.pop();return lower.concat(upper);}
function describeQubit(q){const memberships=DATA.overlays.filter(o=>o.qubits.includes(q.id)).map(o=>o.name);$('qubit-info').textContent=`q${q.id} · (${q.x}, ${q.y}) · ${q.role}`+(memberships.length?' · '+memberships.join(' / '):'');}
function draw(){const target=$('patch-diagram');target.replaceChildren();if(!DATA.qubits.length){target.setAttribute('viewBox','0 0 600 180');target.append(svg('text',{x:30,y:90,fill:'#687b8e'},'No qubit coordinates.'));return;}
const xs=DATA.qubits.map(q=>q.x),ys=DATA.qubits.map(q=>q.y),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
let nearest=Infinity;for(let i=0;i<DATA.qubits.length;i++)for(let j=i+1;j<DATA.qubits.length;j++){const a=DATA.qubits[i],b=DATA.qubits[j],d=Math.hypot(a.x-b.x,a.y-b.y);if(d>0)nearest=Math.min(nearest,d);}if(!Number.isFinite(nearest))nearest=1;
const scale=70/nearest,margin=46,r=17,px=x=>(x-minX)*scale+margin,py=y=>(y-minY)*scale+margin,w=Math.max(2*margin,(maxX-minX)*scale+2*margin),h=Math.max(2*margin,(maxY-minY)*scale+2*margin);
target.setAttribute('viewBox',`0 0 ${w} ${h}`);
if(maxX-minX<=80&&maxY-minY<=80){for(let x=Math.ceil(minX);x<=maxX;x++)target.append(svg('line',{x1:px(x),x2:px(x),y1:margin-16,y2:h-margin+16,stroke:'#eaf0f5','stroke-width':1}));for(let y=Math.ceil(minY);y<=maxY;y++)target.append(svg('line',{y1:py(y),y2:py(y),x1:margin-16,x2:w-margin+16,stroke:'#eaf0f5','stroke-width':1}));}
const support=new Map(selected?selected.support:[]),points=DATA.qubits.filter(q=>support.has(q.id)).map(q=>[px(q.x),py(q.y)]),shape=hull(points),tint=selected?COLORS[selected.basis]:'#8997a5';
if(shape.length>=3)target.append(svg('polygon',{points:shape.map(p=>p.join(',')).join(' '),fill:tint,'fill-opacity':.075,stroke:tint,'stroke-opacity':.24,'stroke-width':2,'stroke-linejoin':'round'}));else if(shape.length===2)target.append(svg('line',{x1:shape[0][0],y1:shape[0][1],x2:shape[1][0],y2:shape[1][1],stroke:tint,'stroke-opacity':.2,'stroke-width':10,'stroke-linecap':'round'}));
for(const q of DATA.qubits){const x=px(q.x),y=py(q.y),factor=support.get(q.id),g=svg('g',{class:'q',tabindex:0,role:'button','aria-label':`Qubit ${q.id}, ${q.role}, coordinates ${q.x}, ${q.y}`});let ring=0;for(const [i,o] of DATA.overlays.entries())if(enabled.has(i)&&o.qubits.includes(q.id)){g.append(svg('circle',{cx:x,cy:y,r:r+5+ring*4,fill:'none',stroke:o.color,'stroke-width':2,opacity:.78}));ring++;}
const attrs={fill:factor?COLORS[factor]:(q.role==='data'?'#fff':'#e8edf3'),stroke:factor?COLORS[factor]:'#91a3b4','stroke-width':1.8};g.append(q.role==='data'?svg('circle',{cx:x,cy:y,r,...attrs}):svg('rect',{x:x-r,y:y-r,width:2*r,height:2*r,rx:4,...attrs}));g.append(svg('text',{x,y:y+.5,'text-anchor':'middle','dominant-baseline':'middle','font-size':12,'font-weight':factor?650:500,fill:factor?'#fff':'#273b4f'},String(q.id)));if(factor)g.append(svg('text',{x:x+r+3,y:y-r-3,'font-size':10,'font-weight':650,fill:COLORS[factor]},factor));const desc=`q${q.id}: (${q.x}, ${q.y}), ${q.role}`+(factor?`, ${factor} support`:'');g.append(svg('title',{},desc));g.addEventListener('click',()=>describeQubit(q));g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();describeQubit(q);}});target.append(g);}
}
function render(){const visible=filtered();if(!visible.includes(selected))selected=visible[0]||null;renderList(visible);draw();if(!selected){$('figure-title').textContent='Patch geometry';$('operator-name').textContent=DATA.operators.length?'No matching operators':'No operators declared';$('category-label').textContent='Geometry only';$('weight').hidden=true;$('pauli').textContent='—';$('detail-meta').textContent=DATA.operators.length?`No ${labels[category].toLowerCase()} match this filter. Clear or change the search to select an operator.`:'The patch has no stabilizer, logical or gauge records.';return;}$('weight').hidden=false;$('figure-title').textContent=selected.name+' · Pauli support';$('category-label').textContent={stabilizer:'Stabilizer generator',logical:'Logical representative',gauge:'Gauge generator'}[selected.category];$('operator-name').textContent=selected.name;$('weight').textContent=`Weight ${selected.weight}`;$('pauli').textContent=selected.pauli_string;let text=`${selected.basis==='mixed'?'Mixed Pauli':selected.basis+'-type'} · sign ${selected.sign===1?'+1':'−1'} · support {${selected.support.map(p=>p[0]).join(', ')}}`;if(selected.syndrome_qubit!==null)text+=` · declared syndrome qubit ${selected.syndrome_qubit}`;if(selected.declared_type&&selected.declared_type!==selected.basis)text+=` · record label: ${selected.declared_type}`;$('detail-meta').textContent=text;}
render();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--factory", required=True, help="Importable module:callable returning a QECPatch.")
    parser.add_argument("--kwargs", default="{}", help="JSON object passed to the factory.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title")
    args = parser.parse_args()
    try:
        module_name, factory_name = args.factory.split(":", 1)
        kwargs = json.loads(args.kwargs)
        if not isinstance(kwargs, dict):
            raise ValueError("--kwargs must be a JSON object.")
        factory = getattr(importlib.import_module(module_name), factory_name)
        path = export_patch_html(factory(**kwargs), args.output, title=args.title)
    except (ValueError, TypeError, AttributeError, ImportError) as exc:
        parser.error(str(exc))
    print(path)


if __name__ == "__main__":
    main()
