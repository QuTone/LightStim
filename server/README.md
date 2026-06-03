# LightStim HTTP Server

A FastAPI HTTP server that exposes LightStim's circuit-construction protocols
over REST endpoints. Pair it with the
[LightStim-front-end](https://github.com/QuTone/LightStim-front-end) web UI
for interactive circuit visualization (DEM 3D, Circuit Timeline, DetSlice
animator).

This server is **optional** — if you only use LightStim from Python (e.g. in
notebooks or scripts), you don't need it. It exists for two use cases:

1. **Local visualization workflow** — power the web UI on your own machine.
2. **Bridge to non-Python clients** — anything that can speak HTTP can drive
   LightStim (curl, JavaScript app, etc.).

## Start

From the repo root, with the venv activated and `pip install -e ".[server]"` done:

```bash
venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 9999
```

Interactive docs (Swagger UI): <http://localhost:9999/docs>

Health check: <http://localhost:9999/> → `{"status":"ok","docs":"/docs"}`

## Endpoints

All endpoints accept a JSON body and return
`{dem, timeline, detslice, circuit_str}`:

| Endpoint | Protocol |
|---|---|
| `POST /api/circuit/memory` | Rotated/unrotated/BB/color/PQRM memory experiments |
| `POST /api/circuit/two-patch-ls` | Two-patch ZZ or XX lattice surgery |
| `POST /api/circuit/transversal-cnot` | Transversal CNOT between two surface patches |
| `POST /api/circuit/cnot-ls` | CNOT via three-patch lattice surgery |
| `POST /api/circuit/ghz` | Three-patch GHZ state preparation |
| `POST /api/circuit/logical-h` | Fold-transversal Hadamard verification |
| `POST /api/circuit/logical-s` | Fold-transversal S-gate verification |
| `POST /api/circuit/multi-patch-ls` | N-patch lattice surgery (N=2–5) |
| `POST /api/circuit/state-injection` | Magic-state injection on rotated SC |
| `POST /api/circuit/bell-teleport` | Bell teleportation (TG and LS variants) |
| `POST /api/circuit/tg-distillation` | Steane distillation with transversal gates |
| `POST /api/circuit/ls-distillation` | Steane distillation with lattice surgery |
| `POST /api/circuit/cross-ls` | Surface–PQRM cross-lattice surgery |

See the auto-generated `/docs` for full parameter schemas.

## Using with the web UI

Two options:

### Option A — Run everything locally
```bash
# Terminal 1: backend (this server)
cd LightStim
venv/bin/uvicorn server.main:app --port 9999

# Terminal 2: frontend
cd ../LightStim-front-end
npm install
npm run dev    # opens http://localhost:8080
```

The frontend's `VITE_API_URL` defaults to `http://localhost:9999`, so it
talks to your local backend out of the box.

### Option B — Use the deployed front-end + your local backend
1. Start this server locally on port 9999.
2. Open the deployed front-end:
   <https://qutone.github.io/LightStim-front-end/>.

The deployed front-end calls `http://localhost:9999` from your browser, so
your local backend is what serves the data. (CORS is open to any origin.)

## What this server does NOT do

- **No simulation / decoding.** It only builds circuits and exports them as
  JSON. Use `lightstim.simulation.SimulationPipeline` for LER estimation —
  that's CPU/GPU intensive and runs out of band.
- **No persistent state.** Each request is independent.
- **No authentication.** Bind to `127.0.0.1` if you don't want it on your
  network.

## Architecture in one diagram

```
┌────────────────────────┐       HTTP       ┌──────────────────────────────┐
│  LightStim-front-end   │ ────────────────► │  server/main.py (this)        │
│  (React in browser)    │  POST /api/...   │  import lightstim.protocols.* │
│  Renders DEM, Timeline │ ◄──────────────── │  → build stim.Circuit         │
│  DetSlice, etc.        │       JSON       │  → export_all() → JSON        │
└────────────────────────┘                  └──────────────────────────────┘
```

The server is a thin wrapper. The actual circuit construction lives in
`lightstim.protocols.*`, and the JSON export lives in
`lightstim.frontend.export.py`.
