"""Decentralized multi-agent coordinator for autonomous V2V communication and distributed collision avoidance."""

import random
from typing import Dict, List, Optional, Tuple
import numpy as np

from core.drone import Drone, DroneState, MissionPriority
from planners.rvo2_3d import ORCAPlanner3D, VelocityConstraint


class V2VBeacon:
    """Vehicle-to-Vehicle broadcast telemetry message."""

    def __init__(
        self,
        drone_id: int,
        pos: np.ndarray,
        vel: np.ndarray,
        priority: MissionPriority,
        soc: float,
        timestamp: float,
    ):
        self.drone_id = drone_id
        self.pos = pos.copy()
        self.vel = vel.copy()
        self.priority = priority
        self.soc = soc
        self.timestamp = timestamp


class DecentralizedCoordinator:
    """Distributed coordination layer executed onboard each independent drone."""

    def __init__(
        self,
        comm_radius_m: float = 250.0,
        packet_loss_rate: float = 0.02, # 2% realistic simulated RF packet loss
        orca_time_horizon: float = 4.5,
    ):
        self.comm_radius_m = comm_radius_m
        self.packet_loss_rate = packet_loss_rate
        self.orca_planner = ORCAPlanner3D(time_horizon=orca_time_horizon)
        
        # Local neighbor perception cache per drone: drone_id -> {neighbor_id: V2VBeacon}
        self.local_neighbor_tables: Dict[int, Dict[int, V2VBeacon]] = {}
        self.v2v_messages_transmitted = 0

    def broadcast_telemetry(self, drones: List[Drone], current_time: float) -> None:
        """Simulates peer-to-peer V2V RF broadcast across local spatial neighborhoods."""
        active_drones = [d for d in drones if d.state == DroneState.CRUISE]
        
        # Reset current cycle neighbor tables
        for d in drones:
            if d.id not in self.local_neighbor_tables:
                self.local_neighbor_tables[d.id] = {}

        for i, sender in enumerate(active_drones):
            beacon = V2VBeacon(
                drone_id=sender.id,
                pos=sender.pos,
                vel=sender.vel,
                priority=sender.priority,
                soc=sender.soc,
                timestamp=current_time,
            )

            for j, receiver in enumerate(active_drones):
                if sender.id == receiver.id:
                    continue

                dist = np.linalg.norm(sender.pos - receiver.pos)
                if dist <= self.comm_radius_m:
                    # Check simulated stochastic RF packet drop
                    if random.random() >= self.packet_loss_rate:
                        self.local_neighbor_tables[receiver.id][sender.id] = beacon
                        self.v2v_messages_transmitted += 1

    def compute_decentralized_velocity(
        self,
        drone: Drone,
        moving_obstacles: List[dict],
        wind_vec: np.ndarray,
        noise_sensitive_zones: List[dict],
    ) -> np.ndarray:
        """Computes collision-free velocity vector autonomously using local peer knowledge."""
        if drone.state != DroneState.CRUISE:
            return np.zeros(3, dtype=np.float64)

        pref_vel = drone.compute_preferred_velocity()
        constraints: List[VelocityConstraint] = []

        # 1. ORCA constraints from locally perceived V2V neighbors
        neighbors = self.local_neighbor_tables.get(drone.id, {})
        for n_id, beacon in neighbors.items():
            # Dynamic reciprocity negotiation:
            # If our drone has emergency priority or critically low battery, we yield less
            reciprocity = 0.5
            if drone.priority.value > beacon.priority.value or drone.soc < 0.20:
                reciprocity = 0.15 # Higher priority yields only 15%
            elif drone.priority.value < beacon.priority.value or beacon.soc < 0.20:
                reciprocity = 0.85 # Lower priority yields 85%

            c = self.orca_planner.compute_orca_halfplane(
                pos_a=drone.pos,
                vel_a=drone.vel,
                radius_a=drone.safety.drone_footprint_radius,
                pos_b=beacon.pos,
                vel_b=beacon.vel,
                radius_b=drone.safety.drone_footprint_radius,
                reciprocity=reciprocity,
                is_obstacle=False,
            )
            constraints.append(c)

        # 2. Constraints from dynamic moving obstacles (e.g. non-cooperative helicopters)
        for obs in moving_obstacles:
            obs_pos = np.array([obs["x"], obs["y"], drone.pos[2]], dtype=np.float64)
            obs_vel = np.array([obs.get("vx", 0.0), obs.get("vy", 0.0), 0.0], dtype=np.float64)
            obs_radius = obs.get("r", 40.0)

            dist = np.linalg.norm(drone.pos[:2] - obs_pos[:2])
            if dist < obs_radius + 150.0:
                c = self.orca_planner.compute_orca_halfplane(
                    pos_a=drone.pos,
                    vel_a=drone.vel,
                    radius_a=drone.safety.drone_footprint_radius,
                    pos_b=obs_pos,
                    vel_b=obs_vel,
                    radius_b=obs_radius,
                    reciprocity=1.0, # Unilateral avoidance of obstacles
                    is_obstacle=True,
                )
                constraints.append(c)

        # 3. Acoustic noise constraint: if approaching ground sensitive zone, clamp descend rate
        for sz in noise_sensitive_zones:
            d_horiz = np.linalg.norm(drone.pos[:2] - np.array([sz["x"], sz["y"]]))
            if d_horiz < sz.get("radius", 80.0):
                # Restrict minimum cruise altitude to reduce ground noise
                if drone.pos[2] < sz.get("min_altitude", 60.0):
                    pref_vel[2] = max(pref_vel[2], 1.5) # Force gentle climb

        # Solve onboard optimization
        cmd_vel = self.orca_planner.solve_linear_program(
            pref_vel=pref_vel,
            constraints=constraints,
            max_speed=drone.kinematics.max_horizontal_speed,
        )

        return cmd_vel
