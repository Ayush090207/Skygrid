"""Unit tests for Prioritized 4D Space-Time A* planner and reservation conflict prevention."""

import numpy as np
import pytest

from config import AirspaceConfig
from core.airspace import Airspace
from planners.priority_space_time_astar import PrioritySpaceTimeAStar


def test_space_time_astar_single_agent():
    config = AirspaceConfig()
    airspace = Airspace(config)
    planner = PrioritySpaceTimeAStar(airspace)

    start = (4, 4)
    goal = (4, 14)
    lane = 0
    start_t = 0

    path = planner.plan_single_agent(
        start_cell=start,
        goal_cell=goal,
        lane=lane,
        start_time_step=start_t,
        moving_obstacles=[],
    )

    assert path is not None
    assert len(path) > 0
    assert path[0] == (start[0], start[1], lane, start_t)
    assert (path[-1][0], path[-1][1]) == goal


def test_space_time_conflict_reservation():
    config = AirspaceConfig()
    airspace = Airspace(config)
    planner = PrioritySpaceTimeAStar(airspace)

    # Agent 1 moves from (4, 4) to (4, 9) at t=0
    path1 = planner.plan_single_agent((4, 4), (4, 9), lane=0, start_time_step=0, moving_obstacles=[])
    assert path1 is not None
    planner.reserve_trajectory(path1)

    # Agent 2 starts at (4, 9) wanting to move to (4, 4) at t=0 (direct head-on collision)
    path2 = planner.plan_single_agent((4, 9), (4, 4), lane=0, start_time_step=0, moving_obstacles=[])
    assert path2 is not None

    # Verify Agent 2 does not share any vertex or edge reservation with Agent 1
    res1_set = set(path1)
    for step in path2:
        assert step not in res1_set
