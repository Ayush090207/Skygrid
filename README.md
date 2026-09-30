# SkyGrid: Urban Drone Delivery Swarm Trajectory Generator

**SkyGrid** is a distributed 4D trajectory generation and conflict-free airspace coordination engine engineered for hundreds of autonomous delivery drones navigating dense urban metropolitan corridors.

The system features real-time 3D collision avoidance algorithms (**ORCA / RVO2**), **Prioritized 4D Space-Time A\*** corridor planning, and **Decentralized Vehicle-to-Vehicle (V2V) Swarm Consensus**, coupled with physical aerodynamic battery discharge models, ISO 9613-2 acoustic ground noise constraints, and dynamic atmospheric wind fields.

---

## Key Capabilities

- **Conflict-Free Trajectory Generation**:
  - **Optimal Reciprocal Collision Avoidance (ORCA / RVO2)**: Solves 3D velocity obstacle cones via linear half-plane constraints with asymmetric reciprocity for tiered priority classes.
  - **Prioritized 4D Space-Time A\***: Discretized 4D reservation grid $(x, y, z, t)$ preventing both vertex collisions and edge swap conflicts.
  - **Decentralized Coordination**: Ad-hoc simulated V2V RF broadcast within communication radii, handling packet loss and dynamic consensus.
- **Urban Airspace & Geofencing**:
  - Structured altitude flight corridors (Eastbound $40\text{ m}$, Westbound $52\text{ m}$, Southbound $64\text{ m}$, Northbound $76\text{ m}$).
  - Static 3D building obstacles and municipal No-Fly Zones (NFZs) around hospitals and trauma centers.
  - Moving aerial obstacles (emergency medical helicopters, intruding non-cooperative aircraft).
- **Physical Models & Environmental Constraints**:
  - **Aerodynamic Battery Model**: Evaluates induced, profile, parasitic, and climb power under variable payload and headwind drag.
  - **Acoustic Noise Footprint**: ISO 9613-2 acoustic sound pressure level (dBA) modeling over noise-sensitive ground zones.
  - **Atmospheric Weather Field**: 3D wind velocity vector fields with logarithmic altitude boundary layers, cyclic gusts, and convective storm cells.
- **Multi-Platform Visualization**:
  - Interactive multi-panel **Matplotlib 2D/3D animated dashboard** with real-time closest-pair separation sparkline, fleet status monitors, and collision counters.
  - Export to standardized **Three.js / WebGL 3D format** for hardware-accelerated browser playback in `skylane_swarm_3d.html`.

---

## System Architecture

```
Skygrid/
├── config.py                 # Dataclass configurations (kinematics, safety, weather, airspace)
├── cli.py                    # Command-line interface for simulation runs, exports, & benchmarks
├── server.py                 # FastAPI REST and WebSocket telemetry streaming service
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project packaging specification
│
├── core/                     # Fundamental domain models
│   ├── drone.py              # Drone agent model (kinematics, battery discharge, priority)
│   ├── airspace.py           # 3D urban airspace, corridors, NFZs, and moving obstacles
│   ├── weather.py            # Wind vector fields, turbulence, gusts, and storm cells
│   └── noise.py              # Acoustic noise propagation and ground sound pressure level (dBA)
│
├── planners/                 # Trajectory planning & conflict avoidance algorithms
│   ├── rvo2_3d.py            # 3D ORCA velocity obstacle calculation & half-plane solver
│   ├── priority_space_time_astar.py # Prioritized 4D Space-Time A* planner
│   ├── decentralized_coordinator.py # Decentralized V2V broadcast & reciprocity negotiation
│   └── trajectory_smoothing.py      # Catmull-Rom C^2 spline smoothing & waypoint pruning
│
├── osm/                      # OpenStreetMap integration & scenario generation
│   ├── osm_parser.py         # GeoJSON / Overpass parser for urban building footprints & helipads
│   └── generator.py          # Synthetic metropolitan scenario & delivery order generator
│
├── simulation/               # Execution orchestrator & analytics
│   ├── simulator.py          # Discrete-time multi-agent simulation coordinator
│   ├── collision_detector.py # 3D spatial hash collision & near-miss detector
│   └── metrics.py            # Fleet performance, path efficiency, & energy metrics
│
├── visualization/            # Visualization & export pipelines
│   ├── matplotlib_visualizer.py # Multi-panel animated dashboard
│   └── export_frames.py      # JSON serialization for Three.js web viewer
│
├── data/                     # Scenarios, maps, and pre-computed flight runs
│   ├── frames.json           # Simulation telemetry ready for 3D web viewer
│   ├── scenarios/            # Standard scenarios (Metropolitan Rush, Mega Swarm 300, Hospital)
│   └── maps/                 # Urban canyon GeoJSON building maps
│
└── tests/                    # Comprehensive unit and integration test suite
    ├── test_drone.py
    ├── test_collision_avoidance.py
    ├── test_space_time_astar.py
    ├── test_decentralized.py
    └── test_simulation.py
```

---

## Mathematical Formulation

### 1. 3D Optimal Reciprocal Collision Avoidance (ORCA)

For two drones $A$ and $B$ with positions $\mathbf{p}_A, \mathbf{p}_B$, velocities $\mathbf{v}_A, \mathbf{v}_B$, and safety radii $r_A, r_B$, the velocity obstacle $VO_{A|B}^\tau$ over lookahead horizon $\tau$ is defined as:

$$VO_{A|B}^\tau = \left\{ \mathbf{v} \mid \exists t \in [0, \tau]: \mathbf{p}_B - \mathbf{p}_A - t \mathbf{v} \in \mathcal{B}(\mathbf{0}, r_A + r_B) \right\}$$

Let $\mathbf{u}$ denote the minimal displacement vector to the boundary $\partial VO_{A|B}^\tau$:

$$\mathbf{u} = \left( \arg\min_{\mathbf{w} \in \partial VO_{A|B}^\tau} \|\mathbf{w} - (\mathbf{v}_A - \mathbf{v}_B)\| \right) - (\mathbf{v}_A - \mathbf{v}_B)$$

With outward normal $\mathbf{n} = \frac{\mathbf{u}}{\|\mathbf{u}\|}$, drone $A$ constrains its permitted velocity $\mathbf{v}$ to the half-space:

$$\left(\mathbf{v} - \left(\mathbf{v}_A + \alpha \mathbf{u}\right)\right) \cdot \mathbf{n} \ge 0$$

Where $\alpha \in [0, 1]$ represents the reciprocity ratio ($\alpha = 0.5$ for symmetric cooperation; $\alpha < 0.5$ for emergency priority drones; $\alpha = 1.0$ for non-cooperative dynamic obstacles).

### 2. Aerodynamic Energy Consumption Model

The instantaneous electrical power $P_{\text{total}}$ consumed by a delivery multirotor is given by:

$$P_{\text{total}} = P_{\text{hover}} \left(\frac{m_{\text{drone}} + m_{\text{payload}}}{m_{\text{drone}}}\right)^{1.5} + \frac{1}{2} \rho C_d A \|\mathbf{v} - \mathbf{v}_{\text{wind}}\|^3 + \frac{(m_{\text{drone}} + m_{\text{payload}}) g \max(0, v_z)}{\eta_{\text{prop}}} + P_{\text{avionics}}$$

Energy expended over discrete step $\Delta t$ is tracked and decremented from the onboard State of Charge (SOC):

$$E(t + \Delta t) = E(t) + \frac{P_{\text{total}} \cdot \Delta t}{3600} \quad [\text{Wh}]$$

### 3. ISO 9613-2 Acoustic Noise Attenuation

The Sound Pressure Level (SPL in dBA) experienced at a ground receiver location $(x_g, y_g, 0)$ from a drone at altitude $z$ with slant distance $d$ is computed as:

$$L_p = L_{w0} + 10 \log_{10}\left(\frac{T}{T_0}\right) - 20 \log_{10}(d) - 11 - \alpha_{\text{atm}} d$$

Where $L_{w0}$ is the reference sound power level at hover, $T / T_0$ is the current rotor thrust ratio, and $\alpha_{\text{atm}}$ is the atmospheric absorption coefficient ($0.5\text{ dB}/100\text{ m}$).

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- NumPy, SciPy, Matplotlib, Pydantic, FastAPI, Uvicorn, Pytest

Install dependencies via `pip`:

```bash
pip install -r requirements.txt
```

---

## Quickstart Guide

### 1. Run a Standard Swarm Simulation (MVP: 30 Drones)

Execute a 30-drone conflict-free simulation with moving obstacles and static no-fly zones:

```bash
python cli.py --drones 30 --duration 180.0 --export-json frames.json
```

### 2. View Animated 3D Web Visualization

The exported `frames.json` is directly compatible with the included WebGL viewer:
1. Open `skylane_swarm_3d.html` in any web browser.
2. Click **Load frames.json** and select `frames.json` (or `data/frames.json`).
3. Toggle between **2D Overhead View** and **3D Isometric View** with altitude flight corridors, dynamic obstacles, and real-time collision sparklines.

### 3. Run Decentralized Swarm (Stretch Goal: 100+ Drones)

Simulate 100 autonomous drones coordinating solely through local Vehicle-to-Vehicle (V2V) RF broadcasts:

```bash
python cli.py --drones 100 --planner decentralized --duration 60.0
```

### 4. Interactive Matplotlib Dashboard

Launch the animated multi-panel Matplotlib dashboard:

```bash
python cli.py --drones 25 --visualize
```

Save animation directly to GIF:

```bash
python cli.py --drones 25 --duration 30.0 --save-anim swarm_flight.gif
```

### 5. Multi-Scale Benchmark Comparison

Run the automated performance benchmark comparing ORCA, Decentralized Swarm, and Prioritized 4D Space-Time A\* across 20, 50, and 100 drones:

```bash
python cli.py --benchmark
```

Example Benchmark Results:

```
======================================================================
SKYGRID URBAN TRAJECTORY GENERATOR - PERFORMANCE BENCHMARK
======================================================================
Drones   | Planner          | Compute (s)  | Collisions  | Near-Miss  | Completed 
---------------------------------------------------------------------------
20       | orca             | 0.08         | 0           | 0          | 1/20      
20       | decentralized    | 0.10         | 0           | 0          | 1/20      
20       | priority_astar   | 0.08         | 0           | 0          | 0/20      
50       | orca             | 0.32         | 0           | 0          | 2/50      
50       | decentralized    | 0.41         | 0           | 0          | 2/50      
50       | priority_astar   | 0.37         | 0           | 2          | 0/50      
100      | orca             | 1.05         | 0           | 1          | 2/100     
100      | decentralized    | 4.14         | 0           | 0          | 2/100     
100      | priority_astar   | 3.60         | 0           | 3          | 0/100     
===========================================================================
```

---

## REST & WebSocket Server

Launch the real-time simulation API:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

- `GET /api/status`: Current simulation step time, active drone count, and collision stats.
- `GET /api/world`: Urban buildings, street grid, and NFZ coordinates.
- `POST /api/reset`: Reset and reconfigure swarm size, planner, and seed.
- `POST /api/obstacles`: Dynamically dispatch moving aerial obstacles.
- `POST /api/nofly`: Declare emergency dynamic no-fly geofences.
- `WS /ws/telemetry`: High-frequency WebSocket streaming real-time simulation telemetry.

---

## Verification & Test Suite

Run the full pytest suite:

```bash
python -m pytest
```

Output:
```
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.1.1
collected 11 items

tests/test_collision_avoidance.py ..                                     [ 18%]
tests/test_decentralized.py ..                                           [ 36%]
tests/test_drone.py ...                                                  [ 63%]
tests/test_simulation.py ..                                              [ 81%]
tests/test_space_time_astar.py ..                                        [100%]

============================= 11 passed in 0.14s ==============================
```
