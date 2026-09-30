"""Discrete-time multi-drone swarm simulation orchestrator."""

from typing import Any, Dict, List, Optional
import numpy as np

from config import SimulationConfig
from core.airspace import Airspace
from core.drone import Drone, DroneState
from core.weather import WeatherField
from planners.rvo2_3d import ORCAPlanner3D, VelocityConstraint
from planners.priority_space_time_astar import PrioritySpaceTimeAStar
from planners.decentralized_coordinator import DecentralizedCoordinator
from simulation.collision_detector import CollisionDetector
from simulation.metrics import MetricsCalculator, SwarmMetrics


class SwarmSimulator:
    """End-to-end discrete event and continuous-time urban swarm trajectory simulator."""

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        airspace: Optional[Airspace] = None,
        drones: Optional[List[Drone]] = None,
    ):
        self.config = config or SimulationConfig()
        self.airspace = airspace or Airspace(self.config.airspace)
        self.weather = WeatherField(self.config.weather)
        self.drones = drones or []
        self.dt = self.config.time_step_sec
        self.current_time = 0.0

        self.collision_detector = CollisionDetector(
            collision_radius_m=self.config.safety.collision_radius_m,
            near_miss_radius_m=self.config.safety.near_miss_radius_m,
        )

        # Planners
        self.orca_planner = ORCAPlanner3D(
            time_horizon=self.config.safety.time_horizon_orca_sec,
            time_step=self.dt,
            default_radius=self.config.safety.drone_footprint_radius,
            max_speed=self.config.kinematics.max_horizontal_speed,
        )
        self.space_time_astar = PrioritySpaceTimeAStar(self.airspace)
        self.decentralized_coordinator = DecentralizedCoordinator(
            comm_radius_m=self.config.safety.communication_radius_m,
            orca_time_horizon=self.config.safety.time_horizon_orca_sec,
        )

        self.telemetry_frames: List[Dict[str, Any]] = []

    def initialize_plans(self) -> None:
        """Pre-computes initial space-time routes if using priority A* planner."""
        if self.config.planner_mode == "priority_astar":
            self.space_time_astar.clear_reservations()
            # Sort drones by priority descending, then departure time
            sorted_drones = sorted(
                self.drones,
                key=lambda d: (-d.priority.value, d.start_time),
            )
            cs = self.airspace.cell_size_m
            obs_data = [obs.to_dict() for obs in self.airspace.dynamic_obstacles]

            for d in sorted_drones:
                s_cell = (int(d.start_pos[0] / cs), int(d.start_pos[1] / cs))
                g_cell = (int(d.goal_pos[0] / cs), int(d.goal_pos[1] / cs))
                start_step = int(d.start_time / 2.0)

                path = self.space_time_astar.plan_single_agent(
                    start_cell=s_cell,
                    goal_cell=g_cell,
                    lane=d.lane_idx,
                    start_time_step=start_step,
                    moving_obstacles=obs_data,
                )

                if path:
                    self.space_time_astar.reserve_trajectory(path)
                    # Convert grid cells to continuous 3D waypoints
                    wps = [
                        (
                            (c[0] + 0.5) * cs,
                            (c[1] + 0.5) * cs,
                            self.airspace.lanes[d.lane_idx],
                        )
                        for c in path
                    ]
                    d.set_route(wps)

    def step(self) -> Dict[str, Any]:
        """Advances simulation by one time step dt."""
        # 1. Dispatch waiting drones whose departure time has arrived
        for d in self.drones:
            if d.state == DroneState.WAITING and self.current_time >= d.start_time:
                d.state = DroneState.CRUISE

        # 2. Advance moving dynamic obstacles and weather disturbances
        self.airspace.step_obstacles(self.dt)
        self.weather.step(self.dt, self.airspace.width_m, self.airspace.height_m)

        obs_dicts = [obs.to_dict() for obs in self.airspace.dynamic_obstacles]
        active_drones = [d for d in self.drones if d.state == DroneState.CRUISE]

        # 3. Trajectory Planning / Velocity Commands
        if self.config.planner_mode == "decentralized":
            # Swarm P2P V2V broadcasting
            self.decentralized_coordinator.broadcast_telemetry(self.drones, self.current_time)
            for d in active_drones:
                wind = self.weather.get_wind_at(d.pos, self.current_time)
                cmd_v = self.decentralized_coordinator.compute_decentralized_velocity(
                    drone=d,
                    moving_obstacles=obs_dicts,
                    wind_vec=wind,
                    noise_sensitive_zones=[],
                )
                d.apply_commanded_velocity(cmd_v, self.dt, wind)

        elif self.config.planner_mode == "orca":
            # Centralized/hybrid multi-agent ORCA solver
            for i, d_a in enumerate(active_drones):
                pref_v = d_a.compute_preferred_velocity()
                constraints: List[VelocityConstraint] = []

                # Drone-to-drone reciprocal avoidance
                for j, d_b in enumerate(active_drones):
                    if i == j:
                        continue
                    # Only enforce ORCA if in nearby flight envelope
                    dist = np.linalg.norm(d_a.pos - d_b.pos)
                    if dist < 80.0:
                        c = self.orca_planner.compute_orca_halfplane(
                            pos_a=d_a.pos,
                            vel_a=d_a.vel,
                            radius_a=d_a.safety.drone_footprint_radius,
                            pos_b=d_b.pos,
                            vel_b=d_b.vel,
                            radius_b=d_b.safety.drone_footprint_radius,
                            reciprocity=0.5,
                            is_obstacle=False,
                        )
                        constraints.append(c)

                # Drone-to-obstacle unilateral avoidance
                for obs in obs_dicts:
                    obs_p = np.array([obs["x"], obs["y"], d_a.pos[2]])
                    obs_v = np.array([obs.get("vx", 0.0), obs.get("vy", 0.0), 0.0])
                    dist = np.linalg.norm(d_a.pos[:2] - obs_p[:2])
                    if dist < obs["r"] + 100.0:
                        c = self.orca_planner.compute_orca_halfplane(
                            pos_a=d_a.pos,
                            vel_a=d_a.vel,
                            radius_a=d_a.safety.drone_footprint_radius,
                            pos_b=obs_p,
                            vel_b=obs_v,
                            radius_b=obs["r"],
                            reciprocity=1.0,
                            is_obstacle=True,
                        )
                        constraints.append(c)

                cmd_v = self.orca_planner.solve_linear_program(
                    pref_vel=pref_v,
                    constraints=constraints,
                    max_speed=d_a.kinematics.max_horizontal_speed,
                )
                wind = self.weather.get_wind_at(d_a.pos, self.current_time)
                d_a.apply_commanded_velocity(cmd_v, self.dt, wind)

        else: # Priority space-time A* strategic waypoint tracking with local reactive safety filter
            for i, d_a in enumerate(active_drones):
                pref_v = d_a.compute_preferred_velocity()
                constraints: List[VelocityConstraint] = []
                for j, d_b in enumerate(active_drones):
                    if i == j:
                        continue
                    dist = np.linalg.norm(d_a.pos - d_b.pos)
                    if dist < 60.0:
                        c = self.orca_planner.compute_orca_halfplane(
                            pos_a=d_a.pos,
                            vel_a=d_a.vel,
                            radius_a=d_a.safety.drone_footprint_radius,
                            pos_b=d_b.pos,
                            vel_b=d_b.vel,
                            radius_b=d_b.safety.drone_footprint_radius,
                            reciprocity=0.5,
                            is_obstacle=False,
                        )
                        constraints.append(c)

                for obs in obs_dicts:
                    obs_p = np.array([obs["x"], obs["y"], d_a.pos[2]])
                    obs_v = np.array([obs.get("vx", 0.0), obs.get("vy", 0.0), 0.0])
                    dist = np.linalg.norm(d_a.pos[:2] - obs_p[:2])
                    if dist < obs["r"] + 80.0:
                        c = self.orca_planner.compute_orca_halfplane(
                            pos_a=d_a.pos,
                            vel_a=d_a.vel,
                            radius_a=d_a.safety.drone_footprint_radius,
                            pos_b=obs_p,
                            vel_b=obs_v,
                            radius_b=obs["r"],
                            reciprocity=1.0,
                            is_obstacle=True,
                        )
                        constraints.append(c)

                cmd_v = self.orca_planner.solve_linear_program(
                    pref_vel=pref_v,
                    constraints=constraints,
                    max_speed=d_a.kinematics.max_horizontal_speed,
                )
                wind = self.weather.get_wind_at(d_a.pos, self.current_time)
                d_a.apply_commanded_velocity(cmd_v, self.dt, wind)

        # 4. Proximity & Conflict Detection
        min_dist, colls, nears, breaches = self.collision_detector.check_frame(
            self.drones, self.airspace, self.current_time
        )

        frame = {
            "t": round(self.current_time, 2),
            "drones": [d.to_dict() for d in self.drones],
            "obstacles": obs_dicts,
            "min": round(min_dist, 2),
            "c": colls,
            "n": nears,
            "v": breaches,
        }
        self.telemetry_frames.append(frame)
        self.current_time += self.dt
        return frame

    def run(self, max_duration: Optional[float] = None) -> SwarmMetrics:
        """Runs the entire simulation until completion or max_duration is reached."""
        duration = max_duration or self.config.total_duration_sec
        self.initialize_plans()

        while self.current_time <= duration:
            self.step()
            # Terminate early if all drones delivered
            if all(d.state in (DroneState.DONE, DroneState.WAITING) for d in self.drones):
                if any(d.state == DroneState.DONE for d in self.drones):
                    break

        v2v_transmitted = (
            self.decentralized_coordinator.v2v_messages_transmitted
            if self.config.planner_mode == "decentralized"
            else 0
        )

        return MetricsCalculator.compute(
            drones=self.drones,
            collisions=self.collision_detector.total_collisions,
            near_misses=self.collision_detector.total_near_misses,
            breaches=self.collision_detector.total_zone_breaches,
            v2v_count=v2v_transmitted,
        )
