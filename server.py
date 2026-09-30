"""FastAPI REST & WebSocket server for real-time swarm telemetry streaming and dynamic obstacle dispatch."""

import asyncio
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import SimulationConfig
from core.airspace import Airspace, DynamicObstacle, NoFlyZone
from osm.generator import UrbanScenarioGenerator
from simulation.simulator import SwarmSimulator

app = FastAPI(
    title="SkyGrid Swarm Trajectory Generator API",
    description="Distributed urban airspace simulation and conflict-free trajectory coordination service.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active simulation instance
SIM_CONFIG = SimulationConfig()
AIRSPACE = Airspace(SIM_CONFIG.airspace)
GENERATOR = UrbanScenarioGenerator(SIM_CONFIG.airspace, seed=42)
DRONES = GENERATOR.generate_drone_fleet(SIM_CONFIG.num_drones, AIRSPACE)
SIMULATOR = SwarmSimulator(config=SIM_CONFIG, airspace=AIRSPACE, drones=DRONES)


class SimulationConfigRequest(BaseModel):
    num_drones: int = 30
    planner_mode: str = "orca"
    duration_sec: float = 120.0
    time_step_sec: float = 0.5
    seed: int = 42


class DynamicObstacleRequest(BaseModel):
    id: str
    x: float
    y: float
    vx: float
    vy: float
    radius_m: float = 40.0


class NoFlyZoneRequest(BaseModel):
    id: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    ceiling_m: float = 120.0
    reason: str = "Ad-hoc Emergency Geofence"


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Returns current simulation state and fleet summary."""
    active_drones = [d.to_dict() for d in SIMULATOR.drones]
    return {
        "status": "online",
        "current_time": SIMULATOR.current_time,
        "drones_count": len(active_drones),
        "collisions": SIMULATOR.collision_detector.total_collisions,
        "near_misses": SIMULATOR.collision_detector.total_near_misses,
        "planner": SIM_CONFIG.planner_mode,
    }


@app.get("/api/world")
def get_world() -> Dict[str, Any]:
    """Returns current airspace geometry, buildings, and no-fly zones."""
    return AIRSPACE.get_world_dict()


@app.post("/api/reset")
def reset_simulation(req: SimulationConfigRequest) -> Dict[str, Any]:
    """Reconfigures and re-initializes swarm simulation."""
    global SIM_CONFIG, AIRSPACE, GENERATOR, DRONES, SIMULATOR
    SIM_CONFIG.num_drones = req.num_drones
    SIM_CONFIG.planner_mode = req.planner_mode
    SIM_CONFIG.total_duration_sec = req.duration_sec
    SIM_CONFIG.time_step_sec = req.time_step_sec
    SIM_CONFIG.random_seed = req.seed

    AIRSPACE = Airspace(SIM_CONFIG.airspace)
    GENERATOR = UrbanScenarioGenerator(SIM_CONFIG.airspace, seed=req.seed)
    DRONES = GENERATOR.generate_drone_fleet(req.num_drones, AIRSPACE)
    SIMULATOR = SwarmSimulator(config=SIM_CONFIG, airspace=AIRSPACE, drones=DRONES)

    return {"message": "Simulation reset successfully", "drones": req.num_drones, "planner": req.planner_mode}


@app.post("/api/obstacles")
def add_moving_obstacle(req: DynamicObstacleRequest) -> Dict[str, Any]:
    """Dynamically introduces an aerial moving obstacle into the airspace."""
    import numpy as np

    obs = DynamicObstacle(
        id=req.id,
        pos=np.array([req.x, req.y, 50.0], dtype=np.float64),
        vel=np.array([req.vx, req.vy, 0.0], dtype=np.float64),
        radius_m=req.radius_m,
    )
    AIRSPACE.dynamic_obstacles.append(obs)
    return {"message": f"Dynamic obstacle {req.id} deployed", "obstacle": obs.to_dict()}


@app.post("/api/nofly")
def add_no_fly_zone(req: NoFlyZoneRequest) -> Dict[str, Any]:
    """Dynamically declares a temporary ground/airspace exclusion zone."""
    nfz = NoFlyZone(
        id=req.id,
        min_x=req.min_x,
        min_y=req.min_y,
        max_x=req.max_x,
        max_y=req.max_y,
        min_z=0.0,
        max_z=req.ceiling_m,
        reason=req.reason,
    )
    AIRSPACE.add_no_fly_zone(nfz)
    return {"message": f"No-fly zone {req.id} enacted", "nfz": req.id}


@app.websocket("/ws/telemetry")
async def websocket_telemetry_stream(websocket: WebSocket):
    """Streams live simulation step frames over high-frequency WebSocket."""
    await websocket.accept()
    try:
        while True:
            frame = SIMULATOR.step()
            await websocket.send_json(frame)
            await asyncio.sleep(SIMULATOR.dt / 2.0) # 2x real-time streaming
    except WebSocketDisconnect:
        pass
