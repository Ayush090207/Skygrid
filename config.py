"""Global configuration and parameters for the SkyGrid swarm trajectory generator."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class KinematicLimits:
    max_horizontal_speed: float = 18.0   # m/s (~65 km/h)
    max_vertical_speed: float = 4.0      # m/s climb/descent
    max_horizontal_accel: float = 5.0    # m/s^2
    max_vertical_accel: float = 2.5      # m/s^2
    cruise_speed: float = 14.0           # m/s nominal cruising speed
    hover_power_watts: float = 280.0     # Power consumption at hover
    avionics_power_watts: float = 35.0   # Power consumption of onboard compute/sensors
    mass_kg: float = 4.5                 # Unloaded drone mass
    max_payload_kg: float = 2.5          # Maximum delivery parcel payload


@dataclass
class SafetyThresholds:
    collision_radius_m: float = 3.0      # Hard physical collision limit
    near_miss_radius_m: float = 8.0      # Alert threshold for safety buffer breach
    drone_footprint_radius: float = 1.8  # Effective bounding cylinder radius
    time_horizon_orca_sec: float = 5.0   # Lookahead time for velocity obstacles
    communication_radius_m: float = 250.0 # V2V peer-to-peer radio range


@dataclass
class AirspaceConfig:
    grid_width_cells: int = 52
    grid_height_cells: int = 34
    cell_size_m: float = 50.0
    lane_altitudes_m: List[float] = field(default_factory=lambda: [40.0, 52.0, 64.0, 76.0])
    # Directions corresponding to altitudes: [East, West, South, North]
    lane_headings_deg: List[float] = field(default_factory=lambda: [0.0, 180.0, 270.0, 90.0])
    noise_sensitive_max_dba: float = 55.0 # Max allowable sound pressure level at ground


@dataclass
class WeatherConfig:
    enabled: bool = True
    base_wind_speed_ms: float = 4.5
    base_wind_dir_deg: float = 45.0      # From North-East
    gust_amplitude_ms: float = 3.0
    gust_period_sec: float = 12.0
    air_density_kg_m3: float = 1.225     # Standard sea-level density


@dataclass
class SimulationConfig:
    time_step_sec: float = 0.5
    total_duration_sec: float = 300.0
    num_drones: int = 30
    random_seed: int = 42
    planner_mode: str = "orca"           # 'orca', 'priority_astar', or 'decentralized'
    kinematics: KinematicLimits = field(default_factory=KinematicLimits)
    safety: SafetyThresholds = field(default_factory=SafetyThresholds)
    airspace: AirspaceConfig = field(default_factory=AirspaceConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
