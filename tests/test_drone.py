"""Unit tests for Drone kinematics, state progression, and battery dynamics."""

import numpy as np
import pytest

from core.drone import Drone, DroneState, MissionPriority
from config import KinematicLimits, SafetyThresholds


def test_drone_initialization():
    drone = Drone(
        drone_id=1,
        start_pos=(100.0, 100.0, 40.0),
        goal_pos=(500.0, 500.0, 40.0),
        start_time=5.0,
        priority=MissionPriority.EXPRESS,
    )
    assert drone.id == 1
    assert drone.state == DroneState.WAITING
    assert drone.soc == 1.0
    assert drone.priority == MissionPriority.EXPRESS
    assert drone.energy_expended_wh == 0.0


def test_drone_preferred_velocity():
    drone = Drone(
        drone_id=2,
        start_pos=(0.0, 0.0, 40.0),
        goal_pos=(100.0, 0.0, 40.0),
    )
    # While waiting, preferred velocity is zero
    assert np.allclose(drone.compute_preferred_velocity(), [0, 0, 0])

    drone.state = DroneState.CRUISE
    pref_vel = drone.compute_preferred_velocity()
    # Preferred velocity should point in +x direction
    assert pref_vel[0] > 0.0
    assert abs(pref_vel[1]) < 1e-4
    assert abs(pref_vel[2]) < 1e-4
    assert np.linalg.norm(pref_vel) <= drone.kinematics.cruise_speed


def test_battery_discharge_under_wind():
    drone = Drone(
        drone_id=3,
        start_pos=(0.0, 0.0, 40.0),
        goal_pos=(200.0, 0.0, 40.0),
        payload_kg=2.0,
    )
    drone.state = DroneState.CRUISE
    cmd_v = np.array([12.0, 0.0, 0.0])
    dt = 1.0
    # Headwind of 5 m/s opposite to drone travel
    headwind = np.array([-5.0, 0.0, 0.0])

    drone.apply_commanded_velocity(cmd_v, dt, headwind)
    assert drone.energy_expended_wh > 0.0
    assert drone.soc < 1.0
    assert drone.pos[0] > 0.0
