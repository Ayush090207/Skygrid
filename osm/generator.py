"""Urban metropolitan environment and delivery demand generator."""

import json
import random
from typing import Dict, List, Tuple
import numpy as np

from config import AirspaceConfig
from core.airspace import Airspace, DynamicObstacle, NoFlyZone
from core.drone import Drone, DroneState, MissionPriority


class UrbanScenarioGenerator:
    """Generates reproducible urban environments, delivery orders, and dynamic hazards."""

    def __init__(self, airspace_config: AirspaceConfig, seed: int = 42):
        self.config = airspace_config
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate_vertiports(self, count: int = 6) -> List[Tuple[float, float, float]]:
        """Generates distribution vertiports at strategic urban arterial hubs."""
        vertiports = []
        cw = self.config.grid_width_cells
        ch = self.config.grid_height_cells
        cs = self.config.cell_size_m

        for _ in range(count):
            # Place vertiports on street intersections
            ix = self.rng.choice([i for i in range(cw) if i % 5 == 4])
            iy = self.rng.choice([j for j in range(ch) if j % 5 == 4])
            vertiports.append(((ix + 0.5) * cs, (iy + 0.5) * cs, 15.0))
        return vertiports

    def generate_drone_fleet(
        self,
        count: int,
        airspace: Airspace,
    ) -> List[Drone]:
        """Generates realistic delivery drone fleet with origins, destinations, and priorities."""
        cw = self.config.grid_width_cells
        ch = self.config.grid_height_cells
        cs = self.config.cell_size_m
        lanes = self.config.lane_altitudes_m

        # Valid street cells
        street_cells = [
            (i, j)
            for i in range(cw)
            for j in range(ch)
            if airspace.street_grid[i, j] == 0 and airspace.nofly_grid[i, j] == 0
        ]

        used_starts = set()
        drones: List[Drone] = []
        for k in range(count):
            # Select start and goal at least 10 blocks apart with unique start positions
            while True:
                s_cell = self.rng.choice(street_cells)
                if len(used_starts) < len(street_cells) and s_cell in used_starts:
                    continue
                g_cell = self.rng.choice(street_cells)
                dist = abs(s_cell[0] - g_cell[0]) + abs(s_cell[1] - g_cell[1])
                if dist >= 10:
                    used_starts.add(s_cell)
                    break

            dx = g_cell[0] - s_cell[0]
            dy = g_cell[1] - s_cell[1]

            # Assign altitude lane based on heading:
            # Lane 0 (40m): Eastbound (+dx)
            # Lane 1 (52m): Westbound (-dx)
            # Lane 2 (64m): Southbound (+dy)
            # Lane 3 (76m): Northbound (-dy)
            if abs(dx) >= abs(dy):
                lane_idx = 0 if dx >= 0 else 1
            else:
                lane_idx = 2 if dy >= 0 else 3

            alt = lanes[lane_idx]
            start_pos = ((s_cell[0] + 0.5) * cs, (s_cell[1] + 0.5) * cs, alt)
            goal_pos = ((g_cell[0] + 0.5) * cs, (g_cell[1] + 0.5) * cs, alt)

            # Assign priority tier
            p_roll = self.rng.random()
            if p_roll < 0.08:
                priority = MissionPriority.EMERGENCY
            elif p_roll < 0.25:
                priority = MissionPriority.EXPRESS
            else:
                priority = MissionPriority.STANDARD

            start_t = self.rng.uniform(0.0, 25.0)

            drone = Drone(
                drone_id=k,
                start_pos=start_pos,
                goal_pos=goal_pos,
                start_time=start_t,
                lane_idx=lane_idx,
                priority=priority,
                payload_kg=self.rng.uniform(0.5, 2.5),
            )
            drones.append(drone)

        return drones

    def export_geojson(self, airspace: Airspace, output_path: str) -> None:
        """Exports the generated urban topology to GeoJSON format."""
        features = []
        cs = self.config.cell_size_m
        
        # Origin lon/lat for conversion
        ref_lon, ref_lat = -122.4194, 37.7749
        m_lat = 111139.0
        m_lon = 111139.0 * np.cos(np.radians(ref_lat))

        def to_geo(x: float, y: float):
            return [ref_lon + (x / m_lon), ref_lat + (y / m_lat)]

        # No Fly Zones
        for nfz in airspace.no_fly_zones:
            poly = [
                to_geo(nfz.min_x, nfz.min_y),
                to_geo(nfz.max_x, nfz.min_y),
                to_geo(nfz.max_x, nfz.max_y),
                to_geo(nfz.min_x, nfz.max_y),
                to_geo(nfz.min_x, nfz.min_y),
            ]
            features.append({
                "type": "Feature",
                "properties": {
                    "id": nfz.id,
                    "type": "no_fly_zone",
                    "reason": nfz.reason,
                    "ceiling_m": nfz.max_z,
                },
                "geometry": {"type": "Polygon", "coordinates": [poly]},
            })

        geojson = {"type": "FeatureCollection", "features": features}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)
