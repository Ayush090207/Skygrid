"""3D Optimal Reciprocal Collision Avoidance (ORCA) and Reciprocal Velocity Obstacles (RVO2)."""

import math
from typing import List, Optional, Tuple
import numpy as np


class VelocityConstraint:
    """Represents a linear half-space constraint: dot(v, normal) >= offset."""

    def __init__(self, normal: np.ndarray, offset: float):
        norm = np.linalg.norm(normal)
        if norm > 1e-8:
            self.normal = normal / norm
            self.offset = offset / norm
        else:
            self.normal = np.array([1.0, 0.0, 0.0])
            self.offset = offset


class ORCAPlanner3D:
    """Optimal Reciprocal Collision Avoidance in 3-dimensional airspace."""

    def __init__(
        self,
        time_horizon: float = 4.0,
        time_step: float = 0.5,
        default_radius: float = 2.5,
        max_speed: float = 18.0,
    ):
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.default_radius = default_radius
        self.max_speed = max_speed

    def compute_orca_halfplane(
        self,
        pos_a: np.ndarray,
        vel_a: np.ndarray,
        radius_a: float,
        pos_b: np.ndarray,
        vel_b: np.ndarray,
        radius_b: float,
        reciprocity: float = 0.5,
        is_obstacle: bool = False,
    ) -> VelocityConstraint:
        """Computes the ORCA linear velocity constraint induced on agent A by agent/obstacle B."""
        rel_pos = pos_b - pos_a
        rel_vel = vel_a - vel_b
        combined_radius = radius_a + radius_b
        dist_sq = float(np.dot(rel_pos, rel_pos))
        dist = math.sqrt(max(1e-6, dist_sq))

        inv_tau = 1.0 / self.time_horizon

        if dist < combined_radius:
            # Overlapping / imminent penetration: emergency push outward along relative position
            w = rel_vel - (rel_pos / self.time_step)
            w_norm = np.linalg.norm(w)
            if w_norm > 1e-6:
                normal = w / w_norm
                u = (combined_radius / self.time_step - w_norm) * normal
            else:
                normal = -rel_pos / dist
                u = (combined_radius - dist) / self.time_step * normal
        else:
            # Standard truncated velocity cone
            w = rel_vel - (rel_pos * inv_tau)
            w_norm = np.linalg.norm(w)
            cutoff_radius = combined_radius * inv_tau

            dot_w_pos = float(np.dot(w, rel_pos))

            if dot_w_pos < 0.0 and (dot_w_pos * dot_w_pos) > (dist_sq * cutoff_radius * cutoff_radius):
                # Projection falls on the cutoff circle
                if w_norm > 1e-6:
                    normal = w / w_norm
                    u = (cutoff_radius - w_norm) * normal
                else:
                    normal = -rel_pos / dist
                    u = np.zeros(3)
            else:
                # Projection falls on the legs of the cone
                # In 3D, find vector perpendicular to relative position
                cross_vec = np.cross(rel_pos, rel_vel)
                cross_norm = np.linalg.norm(cross_vec)
                if cross_norm > 1e-6:
                    perp = np.cross(cross_vec, rel_pos)
                    perp_norm = np.linalg.norm(perp)
                    perp = perp / perp_norm if perp_norm > 1e-6 else np.array([0.0, 1.0, 0.0])
                else:
                    # Arbitrary perpendicular vector to rel_pos
                    if abs(rel_pos[0]) > 1e-4 or abs(rel_pos[1]) > 1e-4:
                        perp = np.array([-rel_pos[1], rel_pos[0], 0.0])
                    else:
                        perp = np.array([0.0, 1.0, 0.0])
                    perp /= np.linalg.norm(perp)

                # Ensure perp aligns with the deflection direction of rel_vel
                if np.dot(rel_vel, perp) < 0.0:
                    perp = -perp

                # Cone opening angle
                sin_alpha = min(1.0, combined_radius / dist)
                cos_alpha = math.sqrt(max(0.0, 1.0 - sin_alpha * sin_alpha))
                
                # Normal pointing outward from the cone boundary
                unit_pos = rel_pos / dist
                normal = (cos_alpha * perp) - (sin_alpha * unit_pos)
                normal /= np.linalg.norm(normal)

                # Closest point on cone surface in direction of normal
                # Distance of rel_vel to the plane containing the cone leg
                penetration = (combined_radius * inv_tau) + float(np.dot(rel_pos * inv_tau - rel_vel, normal))
                if penetration <= 0.0:
                    penetration = max(0.5, combined_radius * inv_tau)
                u = penetration * normal

        # Reciprocity factor: 0.5 for equal cooperators, 1.0 for static/moving obstacles
        alpha = 1.0 if is_obstacle else reciprocity
        point_on_plane = vel_a + (alpha * u)
        offset = float(np.dot(point_on_plane, normal))

        return VelocityConstraint(normal=normal, offset=offset)

    def solve_linear_program(
        self,
        pref_vel: np.ndarray,
        constraints: List[VelocityConstraint],
        max_speed: float,
    ) -> np.ndarray:
        """Solves optimal velocity satisfying linear half-spaces while minimizing deviation from pref_vel."""
        opt_vel = pref_vel.copy()
        speed = np.linalg.norm(opt_vel)
        if speed > max_speed:
            opt_vel = (opt_vel / speed) * max_speed

        # Iteratively check and project onto violating half-planes
        for i, c in enumerate(constraints):
            if np.dot(opt_vel, c.normal) < c.offset:
                # Project opt_vel onto the bounding plane
                correction = (c.offset - np.dot(opt_vel, c.normal)) * c.normal
                candidate = opt_vel + correction

                # Check if candidate satisfies speed limit
                cand_speed = np.linalg.norm(candidate)
                if cand_speed > max_speed:
                    # Project onto intersection with speed sphere
                    # Solve |opt_vel + t * dir|^2 = max_speed^2
                    norm_sq = np.dot(c.normal, c.normal)
                    if abs(c.offset) <= max_speed:
                        rem_radius = math.sqrt(max(0.0, max_speed * max_speed - c.offset * c.offset))
                        # Project remaining horizontal/orthogonal component
                        tangent = candidate - (c.offset * c.normal)
                        t_norm = np.linalg.norm(tangent)
                        if t_norm > 1e-6:
                            candidate = (c.offset * c.normal) + tangent * (rem_radius / t_norm)
                        else:
                            candidate = c.offset * c.normal
                    else:
                        candidate = (c.normal / np.linalg.norm(c.normal)) * max_speed

                # Re-verify with preceding constraints
                valid = True
                for j in range(i):
                    if np.dot(candidate, constraints[j].normal) < constraints[j].offset - 1e-4:
                        valid = False
                        break

                if valid:
                    opt_vel = candidate
                else:
                    # Soft relaxation fallback: choose best velocity minimizing maximum constraint violation
                    opt_vel = self._solve_relaxed(pref_vel, constraints[: i + 1], max_speed)

        return opt_vel

    def _solve_relaxed(
        self,
        pref_vel: np.ndarray,
        constraints: List[VelocityConstraint],
        max_speed: float,
    ) -> np.ndarray:
        """Fallback solver relaxing deepest penetrating constraints when airspace is densely congested."""
        best_vel = pref_vel.copy()
        best_viol = float("inf")

        # Test velocity candidates along normal vectors and unconstrained preferred velocity
        test_candidates = [pref_vel, np.zeros(3)]
        for c in constraints:
            speed = min(max_speed, max(0.0, c.offset))
            test_candidates.append(c.normal * speed)

        for cand in test_candidates:
            cand_speed = np.linalg.norm(cand)
            if cand_speed > max_speed:
                cand = (cand / cand_speed) * max_speed
            
            # Compute total positive penetration depth
            total_viol = 0.0
            for c in constraints:
                diff = c.offset - np.dot(cand, c.normal)
                if diff > 0.0:
                    total_viol += diff * diff

            if total_viol < best_viol:
                best_viol = total_viol
                best_vel = cand

        return best_vel
