"""Exporter converting simulation telemetry into standardized frames.json for Three.js / WebGL visualization."""

import json
from typing import Any, Dict, List
from core.airspace import Airspace


class TelemetryExporter:
    """Exports simulation frames matching the schema expected by skylane_swarm_3d.html."""

    @staticmethod
    def export_to_file(
        frames: List[Dict[str, Any]],
        airspace: Airspace,
        output_filepath: str,
        dt: float = 0.5,
    ) -> None:
        """Serializes world structure and time series telemetry to JSON."""
        # Convert internal drone dictionary format to visualization schema
        serialized_frames = []
        for f in frames:
            drones_list = []
            for d in f.get("drones", []):
                drones_list.append({
                    "id": d["id"],
                    "x": d["x"],
                    "y": d["y"],
                    "z": d.get("z", 40.0),
                    "state": d["state"],
                })

            serialized_frames.append({
                "t": f["t"],
                "drones": drones_list,
                "obstacles": f.get("obstacles", []),
            })

        payload = {
            "meta": {
                "dt": dt,
                "frameCount": len(serialized_frames),
                "generator": "SkyGrid Urban Trajectory Engine",
            },
            "world": airspace.get_world_dict(),
            "frames": serialized_frames,
        }

        with open(output_filepath, "w", encoding="utf-8") as out_f:
            json.dump(payload, out_f, indent=2)
