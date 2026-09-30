"""Simulation execution engine, collision metrics, and trajectory analysis."""

from simulation.simulator import SwarmSimulator
from simulation.collision_detector import CollisionDetector, CollisionEvent
from simulation.metrics import MetricsCalculator, SwarmMetrics

__all__ = [
    "SwarmSimulator",
    "CollisionDetector",
    "CollisionEvent",
    "MetricsCalculator",
    "SwarmMetrics",
]
