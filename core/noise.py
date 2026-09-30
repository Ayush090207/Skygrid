"""Acoustic noise footprint model calculating ground sound pressure level (dBA)."""

import math
from typing import List, Tuple
import numpy as np


class AcousticFootprintModel:
    """Calculates ISO 9613-2 acoustic sound propagation from multi-rotor drones to ground receivers."""

    def __init__(
        self,
        reference_sound_power_dba: float = 78.0, # Source sound power at hover, 1m
        atmospheric_attenuation_db_per_100m: float = 0.5,
    ):
        self.l_w0 = reference_sound_power_dba
        self.alpha_atm = atmospheric_attenuation_db_per_100m / 100.0

    def compute_ground_spl(
        self,
        drone_pos: np.ndarray,
        ground_point: np.ndarray,
        thrust_ratio: float = 1.0,
    ) -> float:
        """Computes Sound Pressure Level (SPL in dBA) experienced at a ground location."""
        # 3D slant distance from drone to ground receiver (z_ground = 0)
        rx = float(drone_pos[0] - ground_point[0])
        ry = float(drone_pos[1] - ground_point[1])
        rz = float(drone_pos[2] - (ground_point[2] if len(ground_point) > 2 else 0.0))
        d = math.sqrt(rx * rx + ry * ry + rz * rz)
        d = max(1.0, d)

        # Source power adjustment for thrust: Lw = Lw0 + 10 * log10(T / T0)
        thrust_factor = 10.0 * math.log10(max(0.5, thrust_ratio))
        l_w = self.l_w0 + thrust_factor

        # Spherical geometric spreading loss: -20 * log10(d) - 11 dB (hemispherical over ground)
        spreading_loss = 20.0 * math.log10(d) + 11.0

        # Atmospheric absorption loss: alpha * d
        atmospheric_loss = self.alpha_atm * d

        spl_dba = l_w - spreading_loss - atmospheric_loss
        return max(20.0, spl_dba) # Background urban ambient floor ~20 dBA

    def compute_cumulative_ground_noise(
        self,
        drone_positions: List[np.ndarray],
        ground_grid_shape: Tuple[int, int],
        cell_size: float,
    ) -> np.ndarray:
        """Computes cumulative energetic decibel sum across a 2D ground grid."""
        nx, ny = ground_grid_shape
        linear_energy = np.zeros((nx, ny), dtype=np.float64)

        for d_pos in drone_positions:
            # Bounding box optimization for noise calculation (~150m cutoff)
            ci = int(d_pos[0] / cell_size)
            cj = int(d_pos[1] / cell_size)
            rad_cells = int(150.0 / cell_size)

            i_min = max(0, ci - rad_cells)
            i_max = min(nx, ci + rad_cells + 1)
            j_min = max(0, cj - rad_cells)
            j_max = min(ny, cj + rad_cells + 1)

            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    rx = (i + 0.5) * cell_size
                    ry = (j + 0.5) * cell_size
                    spl = self.compute_ground_spl(d_pos, np.array([rx, ry, 0.0]))
                    # Acoustic energetic addition: 10^(SPL / 10)
                    linear_energy[i, j] += 10.0 ** (spl / 10.0)

        # Convert back to logarithmic dBA
        noise_map = np.zeros((nx, ny), dtype=np.float64)
        mask = linear_energy > 0
        noise_map[mask] = 10.0 * np.log10(linear_energy[mask])
        noise_map[~mask] = 30.0 # Standard ambient level
        return noise_map
