"""Continuous C2 trajectory smoothing and polynomial waypoint interpolation."""

from typing import List, Tuple
import numpy as np


class TrajectorySmoother:
    """Produces smooth, kinematically feasible spline trajectories from discrete grid routes."""

    @staticmethod
    def smooth_cubic_catmull_rom(
        waypoints: List[Tuple[float, float, float]],
        num_interp_per_segment: int = 10,
    ) -> List[Tuple[float, float, float]]:
        """Applies Catmull-Rom cubic interpolation preserving waypoint accuracy while eliminating sharp corners."""
        if len(waypoints) < 2:
            return waypoints

        pts = [np.array(wp, dtype=np.float64) for wp in waypoints]
        # Duplicate end points for boundary derivatives
        extended = [pts[0]] + pts + [pts[-1]]

        smoothed: List[Tuple[float, float, float]] = []

        for i in range(1, len(extended) - 2):
            p0 = extended[i - 1]
            p1 = extended[i]
            p2 = extended[i + 1]
            p3 = extended[i + 2]

            for s in range(num_interp_per_segment):
                u = s / float(num_interp_per_segment)
                u2 = u * u
                u3 = u2 * u

                # Catmull-Rom spline basis matrix
                pt = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * u
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
                )
                smoothed.append((float(pt[0]), float(pt[1]), float(pt[2])))

        smoothed.append((float(pts[-1][0]), float(pts[-1][1]), float(pts[-1][2])))
        return smoothed

    @staticmethod
    def prune_collinear_waypoints(
        waypoints: List[Tuple[float, float, float]],
        tolerance_deg: float = 1.0,
    ) -> List[Tuple[float, float, float]]:
        """Removes redundant intermediate waypoints along straight corridors to minimize compute overhead."""
        if len(waypoints) <= 2:
            return waypoints

        pruned = [waypoints[0]]
        for i in range(1, len(waypoints) - 1):
            p_prev = np.array(pruned[-1])
            p_curr = np.array(waypoints[i])
            p_next = np.array(waypoints[i + 1])

            v1 = p_curr - p_prev
            v2 = p_next - p_curr

            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)

            if n1 > 1e-4 and n2 > 1e-4:
                cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_theta))
                if angle_deg > tolerance_deg:
                    pruned.append(waypoints[i])
            else:
                pruned.append(waypoints[i])

        pruned.append(waypoints[-1])
        return pruned
