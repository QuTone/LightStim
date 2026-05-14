"""
Generate a syntax-highlighted HTML code block of the LS XX circuit for paper figures.

Builds the circuit with circuit-level noise via NoiseInjector, then assembles a
curated (non-contiguous) snippet showing all four elements:
  - Physical operations (H, CX, M, R)
  - Noise   (DEPOLARIZE1, DEPOLARIZE2, X_ERROR)
  - Detectors
  - Logical observable (OBSERVABLE_INCLUDE)

Output: tests/mwpf_viz_output/ls_xx_code_figure.html
"""

import sys, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment
from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector

from pygments import highlight
from pygments.lexer import RegexLexer, bygroups
from pygments.token import (Token, Keyword, Name, Number, Comment,
                             Punctuation, Text, Generic)
from pygments.formatters import HtmlFormatter

OUTPUT_DIR = Path(__file__).resolve().parent / "mwpf_viz_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Build noisy circuit ───────────────────────────────────────────────────────

exp = TwoPatchLSExperiment(
    patch1_config={"distance": 3},
    patch2_config={"distance": 3},
    offset=(6, 0),
    interaction_type="XX",
    initial_state_patch1="Z",
    initial_state_patch2="Z",
    measure_state_patch1="Z",
    measure_state_patch2="Z",
    rounds=2,
    noise_params=None,
    noise_model=None,
)
clean = exp.build()
nc  = NoiseConfig(p_1q=1e-3, p_2q=1e-3, p_meas=1e-3, p_reset=1e-3, p_idle=1e-3)
inj = NoiseInjector.from_circuit_level(nc, list(range(clean.num_qubits)))
noisy = inj.inject_noise(clean).flattened()

all_lines = str(noisy).split('\n')

def trunc(line, maxq=6):
    """Shorten qubit index lists: keep first maxq tokens after the instruction."""
    m = re.match(r'^(\w[\w_]*(?:\([\d.e+-]+\))?)((?:\s+[\d\s]*))', line)
    if not m:
        return line
    instr = m.group(1)
    rest  = m.group(2).split()
    if len(rest) > maxq:
        rest = rest[:maxq] + ['...']
    return instr + ' ' + ' '.join(rest)

def trunc_rec(line, maxr=4):
    """Shorten rec[-N] lists in DETECTOR / OBSERVABLE_INCLUDE."""
    parts = re.findall(r'rec\[-\d+\]', line)
    if len(parts) <= maxr:
        return line
    head = re.split(r'rec\[-\d+\]', line)[0]
    kept = ' '.join(parts[:maxr]) + ' ...'
    return head + kept

# ── Curated snippet ───────────────────────────────────────────────────────────
# Pick representative lines from different sections; annotate with comments.

def L(i):
    return all_lines[i]

snippet_lines = [
    trunc(all_lines[55]),           # R  (reset)
    trunc(all_lines[56]),           # X_ERROR after reset
    "",
    all_lines[57],                  # TICK[SE_start]
    trunc(all_lines[58]),           # DEPOLARIZE1 (idle before H)
    trunc(all_lines[59]),           # H
    trunc(all_lines[60]),           # DEPOLARIZE1 after H
    all_lines[61],                  # TICK
    trunc(all_lines[62]),           # CX  (first CX layer)
    trunc(all_lines[63]),           # DEPOLARIZE2
    all_lines[64],                  # TICK
    "...",
    trunc(all_lines[77]),           # CX  (last CX layer)
    trunc(all_lines[78]),           # DEPOLARIZE2
    all_lines[79],                  # TICK
    trunc(all_lines[80]),           # H (final)
    trunc(all_lines[81]),           # DEPOLARIZE1
    all_lines[82],                  # TICK
    "",
    trunc(all_lines[83]),           # X_ERROR before M
    trunc(all_lines[84]),           # M
    trunc_rec(all_lines[85]),       # DETECTOR round 1 (single rec)
    trunc_rec(all_lines[86]),
    "...",
    "",
    trunc_rec(all_lines[128]),      # DETECTOR with two recs
    trunc_rec(all_lines[129]),
    "...",
    "",
    trunc_rec(all_lines[278], maxr=5),   # OBSERVABLE_INCLUDE
]

code = '\n'.join(snippet_lines)
print(code)
print(f"\n({len(snippet_lines)} lines)")

# ── Custom Stim lexer ────────────────────────────────────────────────────────

class StimLexer(RegexLexer):
    name = 'Stim'
    tokens = {
        'root': [
            (r'#.*$',                                   Comment.Single),
            (r'OBSERVABLE_INCLUDE',                     Generic.Strong),   # stand-out
            (r'DETECTOR',                               Name.Function),
            (r'(DEPOLARIZE[12]|X_ERROR|Z_ERROR|Y_ERROR|PAULI_CHANNEL_1)',
                                                        Generic.Error),    # noise = red
            (r'(H|CX|CZ|S|S_DAG|SQRT_X|CNOT)\b',      Keyword),
            (r'(M|MX|MZ|MY|MR|MRX|MRZ|R|RX|RZ)\b',    Keyword.Declaration),
            (r'(TICK|SHIFT_COORDS|QUBIT_COORDS)\b',     Name.Builtin),
            (r'rec\[-\d+\]',                            Name.Variable),
            (r'\.\.\.',                                 Comment),
            (r'\([\d.e+-]+\)',                          Number.Float),
            (r'-?\d+',                                  Number.Integer),
            (r'[\[\](),]',                              Punctuation),
            (r'\s+',                                    Text),
        ],
    }

# ── HTML rendering ────────────────────────────────────────────────────────────

formatter = HtmlFormatter(
    style='default',
    full=False,
    linenos=True,
    linenostart=1,
    cssclass='stim-code',
    wrapcode=True,
)

highlighted = highlight(code, StimLexer(), formatter)
css = formatter.get_style_defs('.stim-code')

# Override token colours for clarity
extra_css = """
/* noise instructions → red */
.stim-code .ge  { color: #cc0000; font-weight: bold; }
/* logical observable → purple bold */
.stim-code .gs  { color: #7b00cc; font-weight: bold; font-size: 1.05em; }
/* gate keywords → blue */
.stim-code .k   { color: #0055aa; font-weight: bold; }
/* measurement/reset → teal */
.stim-code .kd  { color: #006666; font-weight: bold; }
/* DETECTOR → green */
.stim-code .nf  { color: #228822; font-weight: bold; }
/* rec[..] → dark orange */
.stim-code .nv  { color: #cc5500; }
/* TICK etc → grey */
.stim-code .nb  { color: #888888; }
/* comments → light grey italic */
.stim-code .c1  { color: #aaaaaa; font-style: italic; }
"""

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>LS XX Circuit — Stim code figure</title>
<style>
  body {{
    background: #ffffff;
    margin: 40px;
    font-family: sans-serif;
  }}
  .stim-code {{
    font-family: 'JetBrains Mono', 'Fira Mono', 'Cascadia Code', 'Consolas', monospace;
    font-size: 14px;
    line-height: 1.55;
    background: #f8f8f8;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 16px 20px;
    display: inline-block;
    min-width: 640px;
  }}
  {css}
  {extra_css}
</style>
</head>
<body>
{highlighted}
</body>
</html>"""

out = OUTPUT_DIR / "ls_xx_code_figure.html"
out.write_text(html)
print(f"\nSaved: {out}")
print("Open in browser → set zoom to taste → screenshot (or Ctrl+S → PDF)")
