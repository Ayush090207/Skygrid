"""Command-line interface for the SkyGrid Urban Swarm Trajectory Generator."""

import argparse
import sys
import time
from typing import Optional

from config import AirspaceConfig, SimulationConfig
from core.airspace import Airspace
from osm.generator import UrbanScenarioGenerator
from simulation.simulator import SwarmSimulator
from visualization.export_frames import TelemetryExporter
from visualization.matplotlib_visualizer import MatplotlibVisualizer


def run_benchmark(drone_counts=(20, 50, 100), duration=40.0) -> None:
    """Executes comparative benchmark between ORCA, Priority A*, and Decentralized Coordinator."""
    print("=" * 70)
    print("SKYGRID URBAN TRAJECTORY GENERATOR - PERFORMANCE BENCHMARK")
    print("=" * 70)

    planners = ["orca", "decentralized", "priority_astar"]

    print(f"{'Drones':<8} | {'Planner':<16} | {'Compute (s)':<12} | {'Collisions':<11} | {'Near-Miss':<10} | {'Completed':<10}")
    print("-" * 75)

    for n in drone_counts:
        for p in planners:
            cfg = SimulationConfig(num_drones=n, total_duration_sec=duration, planner_mode=p, random_seed=42)
            airspace = Airspace(cfg.airspace)
            gen = UrbanScenarioGenerator(cfg.airspace, seed=42)
            drones = gen.generate_drone_fleet(n, airspace)

            sim = SwarmSimulator(config=cfg, airspace=airspace, drones=drones)
            t0 = time.perf_counter()
            metrics = sim.run(max_duration=duration)
            elapsed = time.perf_counter() - t0

            print(
                f"{n:<8} | {p:<16} | {elapsed:<12.2f} | {metrics.total_collisions:<11} | "
                f"{metrics.total_near_misses:<10} | {f'{metrics.completed_drones}/{n}':<10}"
            )
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="SkyGrid: Urban Drone Delivery Swarm Trajectory Generator & Distributed Planner"
    )
    parser.add_argument("--drones", "-n", type=int, default=30, help="Number of delivery drones in fleet (default: 30)")
    parser.add_argument(
        "--planner",
        "-p",
        choices=["orca", "priority_astar", "decentralized"],
        default="orca",
        help="Trajectory planning algorithm (default: orca)",
    )
    parser.add_argument("--duration", "-t", type=float, default=120.0, help="Simulation duration in seconds (default: 120.0)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for repeatable runs (default: 42)")
    parser.add_argument("--export-json", "-e", type=str, default="frames.json", help="Path to export frames.json for 3D web viewer")
    parser.add_argument("--visualize", "-v", action="store_true", help="Launch interactive 2D/3D Matplotlib animation dashboard")
    parser.add_argument("--save-anim", type=str, default=None, help="Save animation to file (.gif or .mp4)")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run benchmark across multiple fleet scales and planners")

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
        return

    print("Initializing SkyGrid Urban Airspace Simulation...")
    print(f"  Fleet size      : {args.drones} drones")
    print(f"  Planner mode    : {args.planner}")
    print(f"  Target duration : {args.duration} s")
    print(f"  Random seed     : {args.seed}")

    config = SimulationConfig(
        num_drones=args.drones,
        total_duration_sec=args.duration,
        planner_mode=args.planner,
        random_seed=args.seed,
    )
    airspace = Airspace(config.airspace)
    gen = UrbanScenarioGenerator(config.airspace, seed=args.seed)
    drones = gen.generate_drone_fleet(args.drones, airspace)

    sim = SwarmSimulator(config=config, airspace=airspace, drones=drones)

    print("Executing distributed trajectory simulation...")
    t_start = time.perf_counter()
    metrics = sim.run(max_duration=args.duration)
    compute_time = time.perf_counter() - t_start

    print(f"Simulation completed in {compute_time:.2f} seconds.")
    print("-" * 50)
    print(metrics.summary_markdown())

    if args.export_json:
        print(f"Exporting {len(sim.telemetry_frames)} frames to {args.export_json}...")
        TelemetryExporter.export_to_file(
            frames=sim.telemetry_frames,
            airspace=airspace,
            output_filepath=args.export_json,
            dt=config.time_step_sec,
        )
        print("Export complete. File is ready to be loaded into skylane_swarm_3d.html")

    if args.visualize or args.save_anim:
        print("Launching visualization dashboard...")
        viz = MatplotlibVisualizer(airspace, sim.telemetry_frames, dt=config.time_step_sec)
        viz.animate(save_path=args.save_anim, show=args.visualize)


if __name__ == "__main__":
    main()
