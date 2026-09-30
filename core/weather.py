"""Weather field simulation modeling 3D wind vectors, turbulence, gusts, and storm cells."""

import math
from typing import Optional, Tuple
import numpy as np

from config import WeatherConfig


class WeatherField:
    def __init__(self, config: Optional[WeatherConfig] = None):
        self.config = config or WeatherConfig()
        
        # Base wind vector components (m/s)
        rad = math.radians(self.config.base_wind_dir_deg)
        # Wind blowing toward heading
        self.base_vx = self.config.base_wind_speed_ms * math.cos(rad)
        self.base_vy = self.config.base_wind_speed_ms * math.sin(rad)
        self.base_vz = 0.0

        # Convective storm cell definition
        self.storm_enabled = True
        self.storm_center = np.array([400.0, 300.0], dtype=np.float64)
        self.storm_velocity = np.array([4.0, 2.5], dtype=np.float64)
        self.storm_radius_m = 180.0
        self.storm_max_downdraft_ms = -3.5

    def step(self, dt: float, max_x: float, max_y: float) -> None:
        """Propagate convective atmospheric disturbances across urban space."""
        if self.storm_enabled:
            self.storm_center += self.storm_velocity * dt
            # Wrap around boundaries to maintain ongoing weather threat
            if self.storm_center[0] > max_x + self.storm_radius_m:
                self.storm_center[0] = -self.storm_radius_m
            if self.storm_center[1] > max_y + self.storm_radius_m:
                self.storm_center[1] = -self.storm_radius_m

    def get_wind_at(self, pos: np.ndarray, t: float) -> np.ndarray:
        """Returns 3D wind velocity vector [vx, vy, vz] at coordinate (x,y,z) and time t."""
        if not self.config.enabled:
            return np.zeros(3, dtype=np.float64)

        # Base wind with altitude logarithmic boundary layer profile
        # Wind speed increases with altitude: v(z) = v_ref * (z / z_ref)^0.2
        alt_factor = max(0.5, (max(10.0, pos[2]) / 40.0) ** 0.22)
        
        # Periodic temporal and spatial gusting
        gust_t = math.sin(2.0 * math.pi * t / self.config.gust_period_sec)
        gust_space = math.cos(0.005 * pos[0] + 0.003 * pos[1])
        gust_mag = self.config.gust_amplitude_ms * (0.6 * gust_t + 0.4 * gust_space)

        vx = (self.base_vx + gust_mag * 0.7) * alt_factor
        vy = (self.base_vy + gust_mag * 0.5) * alt_factor
        vz = 0.0

        # Convective storm cell influence
        if self.storm_enabled:
            d_storm = np.linalg.norm(pos[:2] - self.storm_center)
            if d_storm < self.storm_radius_m:
                intensity = 1.0 - (d_storm / self.storm_radius_m)
                # Severe localized downdraft and radial shear
                vz += self.storm_max_downdraft_ms * intensity
                radial_dir = (pos[:2] - self.storm_center) / max(1.0, d_storm)
                vx += radial_dir[0] * 8.0 * intensity
                vy += radial_dir[1] * 8.0 * intensity

        return np.array([vx, vy, vz], dtype=np.float64)

    def is_in_storm(self, pos: np.ndarray, buffer_m: float = 20.0) -> bool:
        if not self.storm_enabled:
            return False
        dist = np.linalg.norm(pos[:2] - self.storm_center)
        return bool(dist < (self.storm_radius_m + buffer_m))
