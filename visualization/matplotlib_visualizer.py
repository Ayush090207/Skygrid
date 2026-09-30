"""Animated Matplotlib visualization dashboard for 2D/3D swarm flight trajectories."""

import math
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import numpy as np

from core.airspace import Airspace


class MatplotlibVisualizer:
    """Multi-panel animated dashboard with real-time collision monitoring and closest-pair telemetry."""

    def __init__(self, airspace: Airspace, frames: List[Dict[str, Any]], dt: float = 0.5):
        self.airspace = airspace
        self.frames = frames
        self.dt = dt
        self.lane_colors = ["#22d3ee", "#34d399", "#fbbf24", "#e879f9"] # Cyan, Green, Amber, Magenta

    def animate(self, save_path: Optional[str] = None, fps: int = 20, show: bool = True) -> None:
        """Renders animated multi-panel swarm display."""
        if not self.frames:
            print("No frames available to animate.")
            return

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(14, 8), constrained_layout=True)
        gs = fig.add_gridspec(3, 3)

        # Main 2D Airspace Map
        ax_map = fig.add_subplot(gs[:, :2])
        # Live Closest-Pair Distance Chart
        ax_dist = fig.add_subplot(gs[0, 2])
        # Fleet Status Bar Chart
        ax_fleet = fig.add_subplot(gs[1, 2])
        # Safety Counter / Diagnostics
        ax_diag = fig.add_subplot(gs[2, 2])

        ax_map.set_xlim(0, self.airspace.width_m)
        ax_map.set_ylim(0, self.airspace.height_m)
        ax_map.set_aspect("equal")
        ax_map.set_title("Urban Airspace Flight Corridors & Dynamic Conflict Avoidance", fontsize=12, pad=10)
        ax_map.set_xlabel("X (meters)")
        ax_map.set_ylabel("Y (meters)")

        # Render static buildings
        cs = self.airspace.cell_size_m
        for i in range(self.airspace.config.grid_width_cells):
            for j in range(self.airspace.config.grid_height_cells):
                if self.airspace.street_grid[i, j] == 1:
                    rect = Rectangle(
                        (i * cs, j * cs), cs, cs,
                        facecolor="#182131", edgecolor="#0b0f17", alpha=0.9, zorder=1
                    )
                    ax_map.add_patch(rect)

        # Render static No-Fly Zones
        for nfz in self.airspace.no_fly_zones:
            w = nfz.max_x - nfz.min_x
            h = nfz.max_y - nfz.min_y
            rect = Rectangle(
                (nfz.min_x, nfz.min_y), w, h,
                facecolor="#f43f5e", edgecolor="#f43f5e", alpha=0.25, linestyle="--", linewidth=1.5, zorder=2
            )
            ax_map.add_patch(rect)
            ax_map.text(
                nfz.min_x + 5, nfz.min_y + 10, nfz.id,
                color="#f43f5e", fontsize=7, fontweight="bold", zorder=3
            )

        # Dynamic Obstacle patches
        obs_patches = []
        first_frame = self.frames[0]
        for obs in first_frame.get("obstacles", []):
            circle = Circle((obs["x"], obs["y"]), obs["r"], facecolor="#fb923c", edgecolor="#fb923c", alpha=0.35, zorder=4)
            ax_map.add_patch(circle)
            obs_patches.append(circle)

        # Drone scatter points
        drone_scatter = ax_map.scatter([], [], s=36, c=[], edgecolors="#ffffff", linewidths=0.5, zorder=5)
        # Collision alert ring
        alert_scatter = ax_map.scatter([], [], s=120, facecolors="none", edgecolors="#f43f5e", linewidths=2.0, zorder=6)

        # Sparkline Closest Pair Distance
        all_times = [f["t"] for f in self.frames]
        all_mins = [f.get("min", 99.0) for f in self.frames]
        ax_dist.set_xlim(0, max(all_times))
        ax_dist.set_ylim(0, 50.0)
        ax_dist.set_title("Closest Pair Separation (m)", fontsize=10)
        ax_dist.axhline(3.0, color="#f43f5e", linestyle="--", label="Collision (3m)")
        ax_dist.axhline(8.0, color="#fbbf24", linestyle=":", label="Near-miss (8m)")
        dist_line, = ax_dist.plot([], [], color="#34d399", lw=1.5)
        dist_marker, = ax_dist.plot([], [], marker="o", color="#34d399")
        ax_dist.legend(loc="upper right", fontsize=8)

        # Diagnostics Panel
        ax_diag.axis("off")
        diag_text = ax_diag.text(0.05, 0.5, "", fontsize=11, family="monospace", va="center")

        def update(frame_idx):
            f = self.frames[frame_idx]
            drones = f.get("drones", [])
            obstacles = f.get("obstacles", [])

            # Update moving obstacles
            for k, obs in enumerate(obstacles):
                if k < len(obs_patches):
                    obs_patches[k].center = (obs["x"], obs["y"])

            # Filter cruise drones
            cruise_drones = [d for d in drones if d["state"] == "cruise"]
            xs = [d["x"] for d in cruise_drones]
            ys = [d["y"] for d in cruise_drones]
            colors = [self.lane_colors[d.get("lane", 0) % 4] for d in cruise_drones]

            if xs:
                drone_scatter.set_offsets(np.c_[xs, ys])
                drone_scatter.set_color(colors)
            else:
                drone_scatter.set_offsets(np.empty((0, 2)))

            # Distance curve
            sub_times = all_times[: frame_idx + 1]
            sub_mins = all_mins[: frame_idx + 1]
            dist_line.set_data(sub_times, sub_mins)
            if sub_times:
                dist_marker.set_data([sub_times[-1]], [sub_mins[-1]])

            # Fleet state counts
            in_flight = sum(1 for d in drones if d["state"] == "cruise")
            queued = sum(1 for d in drones if d["state"] == "wait")
            delivered = sum(1 for d in drones if d["state"] == "done")

            ax_fleet.clear()
            ax_fleet.set_title("Fleet Mission Status", fontsize=10)
            bars = ax_fleet.bar(["Queued", "In Flight", "Delivered"], [queued, in_flight, delivered],
                                color=["#8a97ab", "#34d399", "#22d3ee"])
            ax_fleet.set_ylim(0, len(drones) + 2)
            for bar in bars:
                h = bar.get_height()
                ax_fleet.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{int(h)}", ha="center", va="bottom", fontsize=9)

            # Diagnostics Text
            colls = f.get("c", 0)
            nears = f.get("n", 0)
            breaches = f.get("v", 0)
            curr_min = f.get("min", 99.0)

            status_str = (
                f"SIMULATION TIME: {f['t']:.1f} s\n"
                f"───────────────────────────\n"
                f"Active In-Flight : {in_flight}\n"
                f"Missions Done    : {delivered}\n"
                f"Closest Pair     : {curr_min:.2f} m\n"
                f"Collisions       : {colls}  {'[CRITICAL]' if colls > 0 else '[OK]'}\n"
                f"Near-Misses      : {nears}\n"
                f"Zone Breaches    : {breaches}\n"
            )
            diag_text.set_text(status_str)

            return [drone_scatter, dist_line, dist_marker, diag_text] + obs_patches

        # Sample every 2nd or 3rd frame for smooth animation playback
        frame_indices = list(range(0, len(self.frames), max(1, len(self.frames) // 180)))
        ani = animation.FuncAnimation(fig, update, frames=frame_indices, interval=1000 // fps, blit=False)

        if save_path:
            print(f"Exporting animation to {save_path}...")
            if save_path.endswith(".gif"):
                ani.save(save_path, writer="pillow", fps=fps)
            else:
                ani.save(save_path, writer="ffmpeg", fps=fps)
            print("Animation exported successfully.")

        if show:
            plt.show()
        else:
            plt.close(fig)
