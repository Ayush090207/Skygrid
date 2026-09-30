"""Prioritized 4D Space-Time A* trajectory planner with vertex and edge conflict reservations."""

import heapq
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from core.airspace import Airspace, DynamicObstacle


class SpaceTimeNode:
    def __init__(self, x: int, y: int, lane: int, time_step: int, g_cost: float, h_cost: float, parent=None):
        self.x = x
        self.y = y
        self.lane = lane
        self.time_step = time_step
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent

    def __lt__(self, other: "SpaceTimeNode") -> bool:
        if abs(self.f_cost - other.f_cost) < 1e-5:
            return self.g_cost > other.g_cost
        return self.f_cost < other.f_cost

    @property
    def key(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.lane, self.time_step)


class PrioritySpaceTimeAStar:
    """Multi-Agent Prioritized Space-Time Path Planner."""

    def __init__(self, airspace: Airspace, max_time_horizon: int = 400):
        self.airspace = airspace
        self.max_time_horizon = max_time_horizon
        self.gw = airspace.config.grid_width_cells
        self.gh = airspace.config.grid_height_cells
        self.lanes = airspace.lanes
        self.cell_size = airspace.cell_size_m

        # 4D reservation structures
        # Vertex reservations: set of (x, y, lane, time_step)
        self.vertex_reservations: Set[Tuple[int, int, int, int]] = set()
        # Edge reservations: set of (from_x, from_y, to_x, to_y, lane, time_step)
        self.edge_reservations: Set[Tuple[int, int, int, int, int, int]] = set()

    def clear_reservations(self) -> None:
        self.vertex_reservations.clear()
        self.edge_reservations.clear()

    def reserve_trajectory(self, path: List[Tuple[int, int, int, int]]) -> None:
        """Commits a planned space-time path into the reservation table."""
        for i, (x, y, lane, t) in enumerate(path):
            self.vertex_reservations.add((x, y, lane, t))
            if i > 0:
                prev_x, prev_y, prev_lane, prev_t = path[i - 1]
                self.edge_reservations.add((prev_x, prev_y, x, y, lane, t))

    def is_blocked(self, x: int, y: int, lane: int, t: int, moving_obstacles: List[dict]) -> bool:
        # Spatial bounds
        if not (0 <= x < self.gw and 0 <= y < self.gh):
            return True

        # Static building check
        if self.airspace.street_grid[x, y] == 1:
            return True

        # Static No-Fly Zone check
        if self.airspace.nofly_grid[x, y] == 1:
            return True

        # Prioritized drone vertex reservation check
        if (x, y, lane, t) in self.vertex_reservations:
            return True

        # Dynamic moving obstacle check at time t
        time_sec = t * 2.0 # Standard discrete interval
        px = (x + 0.5) * self.cell_size
        py = (y + 0.5) * self.cell_size

        for obs in moving_obstacles:
            ox = obs["x"] + obs.get("vx", 0.0) * time_sec
            oy = obs["y"] + obs.get("vy", 0.0) * time_sec
            r = obs["r"]
            dist_sq = (px - ox) ** 2 + (py - oy) ** 2
            if dist_sq < (r + 40.0) ** 2: # 40m safety standoff buffer
                return True

        return False

    def plan_single_agent(
        self,
        start_cell: Tuple[int, int],
        goal_cell: Tuple[int, int],
        lane: int,
        start_time_step: int,
        moving_obstacles: List[dict],
        max_search_depth: int = 250,
    ) -> Optional[List[Tuple[int, int, int, int]]]:
        """A* search through space-time (x, y, lane, t)."""
        sx, sy = start_cell
        gx, gy = goal_cell

        def heuristic(cx: int, cy: int) -> float:
            return float(abs(cx - gx) + abs(cy - gy))

        # Check if start is currently blocked by higher priority agent; advance start time if necessary
        curr_start_t = start_time_step
        while curr_start_t < self.max_time_horizon and self.is_blocked(sx, sy, lane, curr_start_t, moving_obstacles):
            curr_start_t += 1

        if curr_start_t >= self.max_time_horizon:
            return None

        start_node = SpaceTimeNode(
            x=sx,
            y=sy,
            lane=lane,
            time_step=curr_start_t,
            g_cost=0.0,
            h_cost=heuristic(sx, sy),
            parent=None,
        )

        open_set: List[SpaceTimeNode] = []
        heapq.heappush(open_set, start_node)
        visited_costs: Dict[Tuple[int, int, int, int], float] = {start_node.key: 0.0}

        # Allow 4-directional transitions plus wait-in-place
        transitions = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

        while open_set:
            current = heapq.heappop(open_set)

            # Goal reached
            if current.x == gx and current.y == gy:
                # Reconstruct path
                path = []
                curr = current
                while curr:
                    path.append((curr.x, curr.y, curr.lane, curr.time_step))
                    curr = curr.parent
                path.reverse()
                return path

            if current.time_step - curr_start_t >= max_search_depth:
                continue

            next_t = current.time_step + 1

            for dx, dy in transitions:
                nx = current.x + dx
                ny = current.y + dy

                # Check edge collision (prevent head-on swap conflict)
                if (nx, ny, current.x, current.y, lane, next_t) in self.edge_reservations:
                    continue

                if self.is_blocked(nx, ny, lane, next_t, moving_obstacles):
                    continue

                # Transition cost: 1 for movement, 1.1 slight penalty for hovering/waiting to favor progress
                step_cost = 1.0 if (dx != 0 or dy != 0) else 1.1
                g_cost = current.g_cost + step_cost
                key = (nx, ny, lane, next_t)

                if key in visited_costs and g_cost >= visited_costs[key]:
                    continue

                visited_costs[key] = g_cost
                h_cost = heuristic(nx, ny)
                next_node = SpaceTimeNode(
                    x=nx,
                    y=ny,
                    lane=lane,
                    time_step=next_t,
                    g_cost=g_cost,
                    h_cost=h_cost,
                    parent=current,
                )
                heapq.heappush(open_set, next_node)

        return None
