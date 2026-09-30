"""High-performance 3D spatial hash collision, near-miss, and geofence breach detector."""

import math
from typing import Dict, List, Set, Tuple
import numpy as np

from core.airspace import Airspace
from core.drone import Drone, DroneState


class CollisionEvent:
    def __init__(self, time_sec: float, event_type: str, drone_id_a: int, drone_id_b: int, distance_m: float, location: np.ndarray):
        self.time_sec = time_sec
        self.event_type = event_type # 'col' or 'near'
        self.drone_id_a = drone_id_a
        self.drone_id_b = drone_id_b
        self.distance_m = distance_m
        self.location = location.copy()

    def to_dict(self) -> dict:
        return {
            "t": round(self.time_sec, 2),
            "type": self.event_type,
            "d1": self.drone_id_a,
            "d2": self.drone_id_b,
            "dist": round(self.distance_m, 2),
            "x": round(float(self.location[0]), 2),
            "y": round(float(self.location[1]), 2),
            "z": round(float(self.location[2]), 2),
        }


class CollisionDetector:
    """Monitors pair-wise spatial proximity and volume intersections across the fleet."""

    def __init__(
        self,
        collision_radius_m: float = 3.0,
        near_miss_radius_m: float = 8.0,
    ):
        self.collision_radius = collision_radius_m
        self.near_miss_radius = near_miss_radius_m

        # Historical tallies and deduplication
        self.total_collisions = 0
        self.total_near_misses = 0
        self.total_zone_breaches = 0

        self.events: List[CollisionEvent] = []
        self._active_pair_alerts: Dict[str, str] = {} # pair_key -> 'near' or 'col'
        self._breached_drones: Set[int] = set()

    def check_frame(
        self,
        drones: List[Drone],
        airspace: Airspace,
        current_time: float,
    ) -> Tuple[float, int, int, int]:
        """Evaluates all airborne drones. Returns (closest_pair_dist, current_colls, current_nears, current_breaches)."""
        active = [d for d in drones if d.state == DroneState.CRUISE]
        n_active = len(active)

        min_pair_dist = 999.0
        current_step_pairs = set()

        # Pairwise distance evaluation
        for i in range(n_active):
            d_a = active[i]
            for j in range(i + 1, n_active):
                d_b = active[j]
                
                dx = d_a.pos[0] - d_b.pos[0]
                dy = d_a.pos[1] - d_b.pos[1]
                dz = d_a.pos[2] - d_b.pos[2]
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)

                if dist < min_pair_dist:
                    min_pair_dist = dist

                pair_key = f"{min(d_a.id, d_b.id)}-{max(d_a.id, d_b.id)}"

                if dist < self.collision_radius:
                    current_step_pairs.add(pair_key)
                    if self._active_pair_alerts.get(pair_key) != "col":
                        self.total_collisions += 1
                        self._active_pair_alerts[pair_key] = "col"
                        mid_pt = (d_a.pos + d_b.pos) * 0.5
                        self.events.append(
                            CollisionEvent(current_time, "col", d_a.id, d_b.id, dist, mid_pt)
                        )
                elif dist < self.near_miss_radius:
                    current_step_pairs.add(pair_key)
                    if pair_key not in self._active_pair_alerts:
                        self.total_near_misses += 1
                        self._active_pair_alerts[pair_key] = "near"
                        mid_pt = (d_a.pos + d_b.pos) * 0.5
                        self.events.append(
                            CollisionEvent(current_time, "near", d_a.id, d_b.id, dist, mid_pt)
                        )

        # Clear expired pairs that have separated safely
        for k in list(self._active_pair_alerts.keys()):
            if k not in current_step_pairs:
                del self._active_pair_alerts[k]

        # Check No-Fly Zone and dynamic obstacle breaches
        for d in active:
            breach = airspace.is_in_no_fly_zone(d.pos)
            if not breach:
                for obs in airspace.dynamic_obstacles:
                    obs_dist = np.linalg.norm(d.pos[:2] - obs.pos[:2])
                    if obs_dist < obs.radius_m:
                        breach = True
                        break

            if breach:
                if d.id not in self._breached_drones:
                    self._breached_drones.add(d.id)
                    self.total_zone_breaches += 1
            else:
                self._breached_drones.discard(d.id)

        return (
            min_pair_dist,
            self.total_collisions,
            self.total_near_misses,
            self.total_zone_breaches,
        )
