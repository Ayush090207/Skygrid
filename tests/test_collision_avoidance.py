"""Unit tests for 3D ORCA velocity obstacle constraints and reciprocal collision avoidance."""

import numpy as np
import pytest

from planners.rvo2_3d import ORCAPlanner3D, VelocityConstraint


def test_orca_head_on_halfplane():
    planner = ORCAPlanner3D(time_horizon=4.0, default_radius=2.0)
    
    # Two drones flying head-on along X axis
    pos_a = np.array([0.0, 0.0, 40.0])
    vel_a = np.array([10.0, 0.0, 0.0])
    radius_a = 2.0

    pos_b = np.array([50.0, 0.0, 40.0])
    vel_b = np.array([-10.0, 0.0, 0.0])
    radius_b = 2.0

    constraint = planner.compute_orca_halfplane(
        pos_a=pos_a,
        vel_a=vel_a,
        radius_a=radius_a,
        pos_b=pos_b,
        vel_b=vel_b,
        radius_b=radius_b,
        reciprocity=0.5,
    )

    # Current velocity [10, 0, 0] should violate the unconstrained head-on collision condition
    # The normal should exert an evasion vector
    assert np.linalg.norm(constraint.normal) > 0.99
    
    # Solve linear program for agent A
    pref_vel = np.array([10.0, 0.0, 0.0])
    new_vel = planner.solve_linear_program(pref_vel, [constraint], max_speed=15.0)

    # New velocity must satisfy the half-plane constraint
    assert np.dot(new_vel, constraint.normal) >= constraint.offset - 1e-3
    # Agent must deflect orthogonally to avoid head-on crash
    assert abs(new_vel[1]) > 0.1 or abs(new_vel[2]) > 0.1 or new_vel[0] < vel_a[0]


def test_orca_dynamic_obstacle_avoidance():
    planner = ORCAPlanner3D(time_horizon=5.0, default_radius=2.0)
    
    pos_drone = np.array([100.0, 100.0, 50.0])
    vel_drone = np.array([12.0, 0.0, 0.0])

    pos_obs = np.array([140.0, 100.0, 50.0])
    vel_obs = np.array([0.0, 0.0, 0.0])
    radius_obs = 15.0

    constraint = planner.compute_orca_halfplane(
        pos_a=pos_drone,
        vel_a=vel_drone,
        radius_a=2.0,
        pos_b=pos_obs,
        vel_b=vel_obs,
        radius_b=radius_obs,
        reciprocity=1.0, # Unilateral avoidance
        is_obstacle=True,
    )

    new_vel = planner.solve_linear_program(vel_drone, [constraint], max_speed=15.0)
    assert np.dot(new_vel, constraint.normal) >= constraint.offset - 1e-3
