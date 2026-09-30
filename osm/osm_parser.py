"""OpenStreetMap parser converting street corridors, building footprints, and helipads into airspace models."""

import json
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from core.airspace import Airspace, BuildingBlock, NoFlyZone


class OSMUrbanParser:
    """Parses OpenStreetMap GeoJSON / Overpass data into 3D navigable urban airspace."""

    def __init__(self, origin_lat: float = 37.7749, origin_lon: float = -122.4194):
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        # Local tangent plane equirectangular projection approximation (meters per degree)
        self.m_per_deg_lat = 111139.0
        self.m_per_deg_lon = 111139.0 * math.cos(math.radians(origin_lat))

    def geo_to_local_xy(self, lon: float, lat: float) -> Tuple[float, float]:
        """Projects (lon, lat) to local Cartesian coordinates (x, y) in meters."""
        x = (lon - self.origin_lon) * self.m_per_deg_lon
        y = (lat - self.origin_lat) * self.m_per_deg_lat
        return float(x), float(y)

    def parse_geojson(self, geojson_data: Dict[str, Any], airspace: Airspace) -> None:
        """Extracts buildings, highway corridors, and restricted areas from GeoJSON."""
        features = geojson_data.get("features", [])
        
        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            g_type = geom.get("type", "")
            coords = geom.get("coordinates", [])

            # 1. Building footprints
            if "building" in props:
                height = float(props.get("height", props.get("building:levels", 5) * 3.5))
                name = props.get("name", "Urban Building")
                if g_type == "Polygon" and coords:
                    poly_pts = [self.geo_to_local_xy(pt[0], pt[1]) for pt in coords[0]]
                    xs = [p[0] for p in poly_pts]
                    ys = [p[1] for p in poly_pts]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)

                    gx = max(0, int(min_x / airspace.cell_size_m))
                    gy = max(0, int(min_y / airspace.cell_size_m))
                    w_cells = max(1, int(np.ceil((max_x - min_x) / airspace.cell_size_m)))
                    h_cells = max(1, int(np.ceil((max_y - min_y) / airspace.cell_size_m)))

                    if gx < airspace.config.grid_width_cells and gy < airspace.config.grid_height_cells:
                        airspace.buildings.append(
                            BuildingBlock(
                                grid_x=gx,
                                grid_y=gy,
                                width_cells=w_cells,
                                height_cells=h_cells,
                                roof_altitude_m=height,
                                name=name,
                            )
                        )
                        # Mark street grid blocked if below rooftop
                        for bi in range(gx, min(airspace.config.grid_width_cells, gx + w_cells)):
                            for bj in range(gy, min(airspace.config.grid_height_cells, gy + h_cells)):
                                airspace.street_grid[bi, bj] = 1

            # 2. Helipads / Hospitals -> Automatic No-Fly Zones
            if props.get("amenity") == "hospital" or props.get("aeroway") == "helipad":
                name = props.get("name", "Hospital Airspace")
                if g_type == "Polygon" and coords:
                    poly_pts = [self.geo_to_local_xy(pt[0], pt[1]) for pt in coords[0]]
                    xs = [p[0] for p in poly_pts]
                    ys = [p[1] for p in poly_pts]
                    airspace.add_no_fly_zone(
                        NoFlyZone(
                            id=f"NFZ-{props.get('id', len(airspace.no_fly_zones))}",
                            min_x=min(xs),
                            min_y=min(ys),
                            max_x=max(xs),
                            max_y=max(ys),
                            min_z=0.0,
                            max_z=150.0,
                            reason=f"Restricted {name}",
                        )
                    )

    def parse_file(self, filepath: str, airspace: Airspace) -> None:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.parse_geojson(data, airspace)
