# skills/

LightStim Claude Code skills — each skill helps an LLM assist users with a specific API workflow.

## Structure

Each skill is a subdirectory with:
- `SKILL.md` — instructions Claude reads when the skill is triggered
- `scripts/template.py` — complete runnable Python example Claude adapts for the user

```
skills/
├── memory-experiment/       # Build a QEC memory experiment
├── simulate-decode/         # Run simulation pipeline + get LER
├── transversal-cnot/        # Two-patch transversal CNOT gate
├── lattice-surgery-cnot/    # 3-patch lattice surgery CNOT
├── state-injection/         # Inject Z/X/Y logical states
├── custom-noise/            # Configure noise models
├── extend-new-code/         # Add a new QEC code to the library
└── gotchas/                 # Known pitfalls & FAQ — read when debugging
```

## Installing into Claude Code

The skills in this directory are pre-installed for this project via
`.claude/plugins/lightstim/`. They are active whenever you open this repo
in Claude Code — no manual installation needed.

To install them globally (available in any project):

```bash
cp -r skills/* ~/.claude/plugins/lightstim/skills/
```

## Using a skill

Once installed, reference a skill in your prompt or just describe your task —
Claude will trigger the relevant skill automatically. You can also invoke
explicitly in Claude Code:

```
/memory-experiment
/simulate-decode
/extend-new-code
```

## Running the template scripts directly

Each `scripts/template.py` is also a standalone runnable script (from repo root):

```bash
python skills/memory-experiment/scripts/template.py
python skills/extend-new-code/scripts/template.py
```
