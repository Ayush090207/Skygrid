"""Swarm performance metrics, mission completion, and safety analytics."""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from core.drone import Drone, DroneState


@dataclass
class SwarmMetrics:
    total_drones: int
    completed_drones: int
    completion_rate_pct: float
    total_flight_time_sec: float
    mean_flight_time_sec: float
    total_distance_m: float
    mean_path_efficiency_ratio: float
    total_energy_wh: float
    mean_energy_wh: float
    total_collisions: int
    total_near_misses: int
    total_zone_breaches: int
    v2v_messages_count: int

    def summary_markdown(self) -> str:
        return (
            f"| Metric | Value |\n"
            f"|---|---|\n"
            f"| Total Fleet Size | {self.total_drones} drones |\n"
            f"| Deliveries Completed | {self.completed_drones} ({self.completion_rate_pct:.1f}%) |\n"
            f"| Total Collisions | **{self.total_collisions}** |\n"
            f"| Near-Miss Encounters | {self.total_near_misses} |\n"
            f"| NFZ / Obstacle Breaches | {self.total_zone_breaches} |\n"
            f"| Mean Flight Duration | {self.mean_flight_time_sec:.1f} s |\n"
            f"| Path Efficiency Ratio | {self.mean_path_efficiency_ratio:.2f}x Euclidean |\n"
            f"| Total Fleet Energy Expended | {self.total_energy_wh:.2f} Wh |\n"
            f"| Mean Energy per Delivery | {self.mean_energy_wh:.2f} Wh |\n"
            f"| V2V P2P Telemetry Packets | {self.v2v_messages_count} |\n"
        )


class MetricsCalculator:
    @staticmethod
    def compute(
        drones: List[Drone],
        collisions: int,
        near_misses: int,
        breaches: int,
        v2v_count: int = 0,
    ) -> SwarmMetrics:
        total = len(drones)
        completed = sum(1 for d in drones if d.state == DroneState.DONE)
        pct = (completed / max(1, total)) * 100.0

        total_time = sum(d.flight_time_sec for d in drones)
        mean_time = total_time / max(1, completed)

        total_dist = sum(d.distance_traveled_m for d in drones)

        # Efficiency: actual distance / euclidean distance
        eff_ratios = []
        for d in drones:
            direct = float(np.linalg.norm(d.goal_pos - d.start_pos))
            if direct > 1.0 and d.distance_traveled_m > 1.0:
                eff_ratios.append(d.distance_traveled_m / direct)
        mean_eff = float(np.mean(eff_ratios)) if eff_ratios else 1.0

        total_energy = sum(d.energy_expended_wh for d in drones)
        mean_energy = total_energy / max(1, completed)

        return SwarmMetrics(
            total_drones=total,
            completed_drones=completed,
            completion_rate_pct=pct,
            total_flight_time_sec=total_time,
            mean_flight_time_sec=mean_time,
            total_distance_m=total_dist,
            mean_path_efficiency_ratio=mean_eff,
            total_energy_wh=total_energy,
            mean_energy_wh=mean_energy,
            total_collisions=collisions,
            total_near_misses=near_misses,
            total_zone_breaches=breaches,
            v2v_messages_count=v2v_count,
        )
