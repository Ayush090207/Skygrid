"""Autonomous delivery drone agent model with kinematics, battery, and mission state."""

import math
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np

from config import KinematicLimits, SafetyThresholds


class DroneState(Enum):
    WAITING = "wait"
    TAKEOFF = "takeoff"
    CRUISE = "cruise"
    DELIVERY = "delivery"
    DONE = "done"


class MissionPriority(Enum):
    RETURN = 0        # Empty return to hub
    STANDARD = 1      # Regular package
    EXPRESS = 2       # Urgent medicine / rapid courier
    EMERGENCY = 3     # Critical life-safety organ / AED delivery


class Drone:
    def __init__(
        self,
        drone_id: int,
        start_pos: Tuple[float, float, float],
        goal_pos: Tuple[float, float, float],
        start_time: float = 0.0,
        lane_idx: int = 0,
        priority: MissionPriority = MissionPriority.STANDARD,
        payload_kg: float = 1.2,
        battery_capacity_wh: float = 180.0,
        kinematics: Optional[KinematicLimits] = None,
        safety: Optional[SafetyThresholds] = None,
    ):
        self.id = drone_id
        self.start_pos = np.array(start_pos, dtype=np.float64)
        self.goal_pos = np.array(goal_pos, dtype=np.float64)
        self.pos = np.array(start_pos, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.pref_vel = np.zeros(3, dtype=np.float64)
        
        self.start_time = start_time
        self.lane_idx = lane_idx
        self.priority = priority
        self.payload_kg = payload_kg
        self.state = DroneState.WAITING
        
        self.kinematics = kinematics or KinematicLimits()
        self.safety = safety or SafetyThresholds()
        
        # Energy & Battery Tracking
        self.battery_capacity_wh = battery_capacity_wh
        self.energy_expended_wh = 0.0
        self.soc = 1.0 # 100% initial charge
        
        # Mission Navigation Waypoints
        self.waypoints: List[np.ndarray] = [self.goal_pos.copy()]
        self.current_wp_idx = 0
        self.distance_traveled_m = 0.0
        self.flight_time_sec = 0.0
        
        # Telemetry history for trajectory inspection
        self.trajectory_history: List[Tuple[float, float, float, float]] = [] # (t, x, y, z)

    @property
    def x(self) -> float:
        return float(self.pos[0])

    @property
    def y(self) -> float:
        return float(self.pos[1])

    @property
    def z(self) -> float:
        return float(self.pos[2])

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    def set_route(self, route_pts: List[Tuple[float, float, float]]) -> None:
        """Assign an active sequential waypoint path."""
        self.waypoints = [np.array(pt, dtype=np.float64) for pt in route_pts]
        self.current_wp_idx = 0

    def compute_preferred_velocity(self) -> np.ndarray:
        """Calculates preferred unconstrained velocity vector pointing toward active waypoint."""
        if self.state in (DroneState.WAITING, DroneState.DONE):
            self.pref_vel = np.zeros(3, dtype=np.float64)
            return self.pref_vel

        if self.current_wp_idx >= len(self.waypoints):
            target = self.goal_pos
        else:
            target = self.waypoints[self.current_wp_idx]

        disp = target - self.pos
        dist = np.linalg.norm(disp)

        if dist < 2.0:
            if self.current_wp_idx < len(self.waypoints) - 1:
                self.current_wp_idx += 1
                target = self.waypoints[self.current_wp_idx]
                disp = target - self.pos
                dist = np.linalg.norm(disp)
            else:
                self.state = DroneState.DONE
                self.pref_vel = np.zeros(3, dtype=np.float64)
                return self.pref_vel

        # Desired horizontal cruising speed
        desired_speed = min(self.kinematics.cruise_speed, dist)
        if dist > 1e-4:
            unit_dir = disp / dist
            self.pref_vel = unit_dir * desired_speed
            # Limit vertical climb/descent
            self.pref_vel[2] = np.clip(
                self.pref_vel[2],
                -self.kinematics.max_vertical_speed,
                self.kinematics.max_vertical_speed,
            )
        else:
            self.pref_vel = np.zeros(3, dtype=np.float64)

        return self.pref_vel

    def apply_commanded_velocity(self, cmd_vel: np.ndarray, dt: float, wind_vec: np.ndarray) -> None:
        """Applies commanded velocity respecting kinematic acceleration bounds and wind influence."""
        if self.state == DroneState.WAITING:
            return

        # Acceleration limiting
        accel = (cmd_vel - self.vel) / dt
        horiz_accel = np.linalg.norm(accel[:2])
        if horiz_accel > self.kinematics.max_horizontal_accel:
            scale = self.kinematics.max_horizontal_accel / horiz_accel
            accel[0] *= scale
            accel[1] *= scale

        if abs(accel[2]) > self.kinematics.max_vertical_accel:
            accel[2] = np.sign(accel[2]) * self.kinematics.max_vertical_accel

        self.vel = self.vel + accel * dt

        # Clamp max speeds
        horiz_speed = np.linalg.norm(self.vel[:2])
        if horiz_speed > self.kinematics.max_horizontal_speed:
            self.vel[:2] = (self.vel[:2] / horiz_speed) * self.kinematics.max_horizontal_speed

        self.vel[2] = np.clip(
            self.vel[2],
            -self.kinematics.max_vertical_speed,
            self.kinematics.max_vertical_speed,
        )

        step_displacement = self.vel * dt
        self.pos += step_displacement
        step_dist = float(np.linalg.norm(step_displacement))
        self.distance_traveled_m += step_dist
        self.flight_time_sec += dt

        # Update battery consumption based on aerodynamic aerodynamic power
        self._update_energy(step_dist, dt, wind_vec)

    def _update_energy(self, step_dist: float, dt: float, wind_vec: np.ndarray) -> None:
        """Physical aerodynamic power consumption model."""
        total_mass = self.kinematics.mass_kg + (self.payload_kg if self.state != DroneState.DONE else 0.0)
        mass_ratio = total_mass / self.kinematics.mass_kg
        
        # Airspeed vector relative to local wind
        airspeed_vec = self.vel - wind_vec
        airspeed = float(np.linalg.norm(airspeed_vec))
        
        # Induced and profile power:
        # P_induced proportional to mass^1.5 / max(airspeed, hover_v_induced)
        # P_parasitic proportional to 0.5 * rho * Cd * Area * airspeed^3
        hover_p = self.kinematics.hover_power_watts * (mass_ratio ** 1.5)
        parasite_p = 0.5 * 1.225 * 0.08 * (airspeed ** 3) # Equivalent flat plate drag area ~0.08 m^2
        climb_p = total_mass * 9.81 * max(0.0, float(self.vel[2])) / 0.75 # ~75% propulsive efficiency
        
        power_watts = hover_p + parasite_p + climb_p + self.kinematics.avionics_power_watts
        energy_step_wh = (power_watts * dt) / 3600.0
        
        self.energy_expended_wh += energy_step_wh
        self.soc = max(0.0, 1.0 - (self.energy_expended_wh / self.battery_capacity_wh))

    def to_dict(self) -> dict:
        """Serializes current drone telemetry for visualization."""
        return {
            "id": self.id,
            "x": round(float(self.pos[0]), 2),
            "y": round(float(self.pos[1]), 2),
            "z": round(float(self.pos[2]), 2),
            "vx": round(float(self.vel[0]), 2),
            "vy": round(float(self.vel[1]), 2),
            "vz": round(float(self.vel[2]), 2),
            "state": self.state.value,
            "soc": round(self.soc, 3),
            "lane": self.lane_idx,
            "priority": self.priority.name,
        }
