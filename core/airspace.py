"""Urban 3D airspace model representing street corridors, buildings, dynamic obstacles, and NFZs."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from config import AirspaceConfig


@dataclass
class BuildingBlock:
    grid_x: int
    grid_y: int
    width_cells: int
    height_cells: int
    roof_altitude_m: float
    name: str = "Commercial Skyscraper"


@dataclass
class NoFlyZone:
    id: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    min_z: float = 0.0
    max_z: float = 120.0
    reason: str = "Hospital Heliport Exclusion Buffer"

    def contains(self, pos: np.ndarray, buffer_m: float = 0.0) -> bool:
        return (
            (self.min_x - buffer_m <= pos[0] <= self.max_x + buffer_m)
            and (self.min_y - buffer_m <= pos[1] <= self.max_y + buffer_m)
            and (self.min_z <= pos[2] <= self.max_z)
        )


@dataclass
class DynamicObstacle:
    id: str
    pos: np.ndarray
    vel: np.ndarray
    radius_m: float
    altitude_min_m: float = 20.0
    altitude_max_m: float = 90.0

    def step(self, dt: float, bounds_x: float, bounds_y: float) -> None:
        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        
        # Bounce smoothly off boundary perimeters
        if self.pos[0] - self.radius_m < 0:
            self.pos[0] = self.radius_m
            self.vel[0] *= -1.0
        elif self.pos[0] + self.radius_m > bounds_x:
            self.pos[0] = bounds_x - self.radius_m
            self.vel[0] *= -1.0

        if self.pos[1] - self.radius_m < 0:
            self.pos[1] = self.radius_m
            self.vel[1] *= -1.0
        elif self.pos[1] + self.radius_m > bounds_y:
            self.pos[1] = bounds_y - self.radius_m
            self.vel[1] *= -1.0

    def to_dict(self) -> dict:
        return {
            "x": round(float(self.pos[0]), 2),
            "y": round(float(self.pos[1]), 2),
            "r": round(float(self.radius_m), 2),
            "vx": round(float(self.vel[0]), 2),
            "vy": round(float(self.vel[1]), 2),
        }


class Airspace:
    def __init__(self, config: Optional[AirspaceConfig] = None):
        self.config = config or AirspaceConfig()
        self.width_m = self.config.grid_width_cells * self.config.cell_size_m
        self.height_m = self.config.grid_height_cells * self.config.cell_size_m
        self.cell_size_m = self.config.cell_size_m

        self.lanes = self.config.lane_altitudes_m
        self.headings = self.config.lane_headings_deg

        self.buildings: List[BuildingBlock] = []
        self.no_fly_zones: List[NoFlyZone] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []
        
        # Grid arrays for discrete path searches: 0 = free corridor, 1 = building blocked
        self.street_grid = np.zeros(
            (self.config.grid_width_cells, self.config.grid_height_cells), dtype=np.uint8
        )
        self.nofly_grid = np.zeros(
            (self.config.grid_width_cells, self.config.grid_height_cells), dtype=np.uint8
        )

        self._build_default_urban_layout()

    def _build_default_urban_layout(self) -> None:
        """Constructs an authentic urban grid with arterial streets and building plots."""
        # Street layout: every 5th cell is a street corridor (both horizontal and vertical)
        for i in range(self.config.grid_width_cells):
            for j in range(self.config.grid_height_cells):
                is_street = (i % 5 == 4) or (j % 5 == 4)
                if not is_street:
                    self.street_grid[i, j] = 1 # Blocked by building
                else:
                    self.street_grid[i, j] = 0 # Sky corridor open

        # Standard Urban No-Fly Zones (e.g. City Central Hospital, Government Administration)
        self.add_no_fly_zone(
            NoFlyZone(
                id="NFZ-MED-01",
                min_x=8 * self.cell_size_m,
                min_y=6 * self.cell_size_m,
                max_x=12 * self.cell_size_m,
                max_y=9 * self.cell_size_m,
                min_z=0.0,
                max_z=120.0,
                reason="Metropolitan Trauma Center LifeFlight Approach",
            )
        )
        self.add_no_fly_zone(
            NoFlyZone(
                id="NFZ-GOV-02",
                min_x=26 * self.cell_size_m,
                min_y=15 * self.cell_size_m,
                max_x=31 * self.cell_size_m,
                max_y=19 * self.cell_size_m,
                min_z=0.0,
                max_z=150.0,
                reason="Civic Hall & Municipal Court Airspace",
            )
        )

        # Standard moving aerial obstacles (e.g. Police / Emergency Helicopter, Survey Blimp)
        self.dynamic_obstacles.append(
            DynamicObstacle(
                id="OBS-MEDIVAC-1",
                pos=np.array([300.0, 250.0, 55.0], dtype=np.float64),
                vel=np.array([16.0, 9.0, 0.0], dtype=np.float64),
                radius_m=45.0,
            )
        )
        self.dynamic_obstacles.append(
            DynamicObstacle(
                id="OBS-SURVEY-2",
                pos=np.array([1500.0, 1000.0, 65.0], dtype=np.float64),
                vel=np.array([-12.0, -14.0, 0.0], dtype=np.float64),
                radius_m=45.0,
            )
        )

    def add_no_fly_zone(self, nfz: NoFlyZone) -> None:
        self.no_fly_zones.append(nfz)
        i0 = max(0, int(nfz.min_x / self.cell_size_m))
        i1 = min(self.config.grid_width_cells, int(np.ceil(nfz.max_x / self.cell_size_m)))
        j0 = max(0, int(nfz.min_y / self.cell_size_m))
        j1 = min(self.config.grid_height_cells, int(np.ceil(nfz.max_y / self.cell_size_m)))
        for i in range(i0, i1):
            for j in range(j0, j1):
                self.nofly_grid[i, j] = 1

    def is_in_no_fly_zone(self, pos: np.ndarray, buffer_m: float = 0.0) -> bool:
        for nfz in self.no_fly_zones:
            if nfz.contains(pos, buffer_m):
                return True
        return False

    def is_building_collision(self, pos: np.ndarray) -> bool:
        i = int(pos[0] / self.cell_size_m)
        j = int(pos[1] / self.cell_size_m)
        if 0 <= i < self.config.grid_width_cells and 0 <= j < self.config.grid_height_cells:
            # If cell is building and drone is below rooftop clearance (typically 35m)
            if self.street_grid[i, j] == 1 and pos[2] < 35.0:
                return True
        return False

    def step_obstacles(self, dt: float) -> None:
        for obs in self.dynamic_obstacles:
            obs.step(dt, self.width_m, self.height_m)

    def get_world_dict(self) -> dict:
        """Exports world structure compatible with Three.js web viewer."""
        blocked_cells = []
        for i in range(self.config.grid_width_cells):
            for j in range(self.config.grid_height_cells):
                if self.street_grid[i, j] == 1:
                    blocked_cells.append([i, j])

        nofly_cells = []
        for i in range(self.config.grid_width_cells):
            for j in range(self.config.grid_height_cells):
                if self.nofly_grid[i, j] == 1:
                    nofly_cells.append([i, j])

        return {
            "width": self.width_m,
            "height": self.height_m,
            "cell": self.cell_size_m,
            "blocked": blocked_cells,
            "nofly": nofly_cells,
        }
