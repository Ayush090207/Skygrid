"""Unit tests for Decentralized Swarm Coordinator and V2V communication layer."""

import numpy as np
import pytest

from core.drone import Drone, DroneState, MissionPriority
from planners.decentralized_coordinator import DecentralizedCoordinator


def test_v2v_broadcast_filtering_by_distance():
    coord = DecentralizedCoordinator(comm_radius_m=100.0, packet_loss_rate=0.0)

    drone_a = Drone(1, (0.0, 0.0, 40.0), (200.0, 0.0, 40.0))
    drone_b = Drone(2, (50.0, 0.0, 40.0), (200.0, 0.0, 40.0))   # Distance = 50m (in range)
    drone_c = Drone(3, (500.0, 0.0, 40.0), (700.0, 0.0, 40.0)) # Distance = 500m (out of range)

    drone_a.state = DroneState.CRUISE
    drone_b.state = DroneState.CRUISE
    drone_c.state = DroneState.CRUISE

    fleet = [drone_a, drone_b, drone_c]
    coord.broadcast_telemetry(fleet, current_time=1.0)

    # Drone A should see Drone B, but not Drone C
    neighbors_a = coord.local_neighbor_tables.get(drone_a.id, {})
    assert drone_b.id in neighbors_a
    assert drone_c.id not in neighbors_a


def test_asymmetric_priority_reciprocity():
    coord = DecentralizedCoordinator(comm_radius_m=150.0, packet_loss_rate=0.0)

    emergency_drone = Drone(10, (100.0, 100.0, 40.0), (300.0, 100.0, 40.0), priority=MissionPriority.EMERGENCY)
    standard_drone = Drone(11, (150.0, 100.0, 40.0), (50.0, 100.0, 40.0), priority=MissionPriority.STANDARD)

    emergency_drone.state = DroneState.CRUISE
    standard_drone.state = DroneState.CRUISE

    coord.broadcast_telemetry([emergency_drone, standard_drone], current_time=2.0)

    # Emergency drone velocity calculation
    v_emerg = coord.compute_decentralized_velocity(
        emergency_drone, moving_obstacles=[], wind_vec=np.zeros(3), noise_sensitive_zones=[]
    )
    assert np.linalg.norm(v_emerg) > 0.0
