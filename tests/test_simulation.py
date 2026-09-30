"""End-to-end integration tests for multi-drone swarm simulation execution."""

import pytest

from config import SimulationConfig
from core.airspace import Airspace
from osm.generator import UrbanScenarioGenerator
from simulation.simulator import SwarmSimulator


def test_swarm_simulation_execution():
    config = SimulationConfig(
        num_drones=10,
        total_duration_sec=30.0,
        time_step_sec=0.5,
        planner_mode="orca",
        random_seed=42,
    )
    airspace = Airspace(config.airspace)
    gen = UrbanScenarioGenerator(config.airspace, seed=42)
    drones = gen.generate_drone_fleet(10, airspace)

    sim = SwarmSimulator(config=config, airspace=airspace, drones=drones)
    metrics = sim.run(max_duration=30.0)

    assert len(sim.telemetry_frames) > 0
    assert metrics.total_drones == 10
    # ORCA should keep flights safe
    assert metrics.total_collisions == 0


def test_simulation_telemetry_frame_structure():
    config = SimulationConfig(
        num_drones=5,
        total_duration_sec=5.0,
        time_step_sec=0.5,
        planner_mode="orca",
        random_seed=123,
    )
    airspace = Airspace(config.airspace)
    gen = UrbanScenarioGenerator(config.airspace, seed=123)
    drones = gen.generate_drone_fleet(5, airspace)

    sim = SwarmSimulator(config=config, airspace=airspace, drones=drones)
    frame = sim.step()

    assert "t" in frame
    assert "drones" in frame
    assert "obstacles" in frame
    assert "min" in frame
    assert "c" in frame
    assert "n" in frame
    assert len(frame["drones"]) == 5
