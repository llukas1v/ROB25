#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-08-21
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing planar manipulator."""

#import matplotlib.pyplot as plt


from __future__ import annotations
from copy import copy, deepcopy
from math import isclose
import numpy as np
from numpy.typing import ArrayLike
from shapely import MultiPolygon, LineString, MultiLineString

from robotics_toolbox.core import SE2, SE3
from robotics_toolbox.core.so2 import SO2
from robotics_toolbox.robots.robot_base import RobotBase


class PlanarManipulator(RobotBase):
    def __init__(
        self,
        link_parameters: ArrayLike | None = None,
        structure: list[str] | str | None = None,
        base_pose: SE2 | None = None,
        gripper_length: float = 0.2,
    ) -> None:
        """
        Creates a planar manipulator composed by rotational and prismatic joints.

        The manipulator kinematics is defined by following kinematics chain:
         T_flange = (T_base) T(q_0) T(q_1) ... T_n(q_n),
        where
         T_i describes the pose of the next link w.r.t. the previous link computed as:
         T_i = R(q_i) Tx(l_i) if joint is revolute,
         T_i = R(l_i) Tx(q_i) if joint is prismatic,
        with
         l_i is taken from @param link_parameters;
         type of joint is defined by the @param structure.

        Args:
            link_parameters: either the lengths of links attached to revolute joints
             in [m] or initial rotation of prismatic joint [rad].
            structure: sequence of joint types, either R or P, [R]*n by default
            base_pose: mounting of the robot, identity by default
            gripper_length: length of the gripper measured from the flange
        """
        super().__init__()
        self.link_parameters: np.ndarray = np.asarray(
            [0.5] * 3 if link_parameters is None else link_parameters
        )
        n = len(self.link_parameters)
        self.base_pose = SE2() if base_pose is None else base_pose
        self.structure = ["R"] * n if structure is None else structure
        assert len(self.structure) == len(self.link_parameters)
        self.gripper_length = gripper_length

        # Robot configuration:
        self.q = np.array([np.pi / 8] * n)
        self.gripper_opening = 0.2

        # Configuration space
        self.q_min = np.array([-np.pi] * n)
        self.q_max = np.array([np.pi] * n)

        # Additional obstacles for collision checking function
        self.obstacles: MultiPolygon | None = None

    @property
    def dof(self):
        """Return number of degrees of freedom."""
        return len(self.q)

    def sample_configuration(self):
        """Sample robot configuration inside the configuration space. Will change
        internal state."""
        return np.random.uniform(self.q_min, self.q_max)

    def set_configuration(self, configuration: np.ndarray | SE2 | SE3):
        """Set configuration of the robot, return self for chaining."""
        self.q = configuration
        return self

    def configuration(self) -> np.ndarray | SE2 | SE3:
        """Get the robot configuration."""
        return self.q

    def flange_pose(self) -> SE2:
        """Return the pose of the flange in the reference frame."""
        # todo HW02: implement fk for the flange


        fl = self.base_pose

        for i in range(self.dof):
            if self.structure[i] == 'R':
                n = SE2( rotation= self.q[i])*SE2(translation= [self.link_parameters[i],0])
            else:
                n = SE2(  rotation= self.link_parameters[i])*SE2(translation= [self.q[i],0] )
            fl *= n
        return fl

    def fk_all_links(self) -> list[SE2]:
        """Compute FK for frames that are attached to the links of the robot.
        The first frame is base_frame, the next frames are described in the constructor.
        """
        # todo HW02: implement fk
        frames = []
        frames.append(self.base_pose)
        for i in range(self.dof):
            if self.structure[i] == 'R':
                n = SE2( rotation= self.q[i])*SE2(translation= [self.link_parameters[i],0])
            else:
                n = SE2(  rotation= self.link_parameters[i])*SE2(translation= [self.q[i],0] )
            frames.append(frames[i]*n)
        return frames

    def _gripper_lines(self, flange: SE2):
        """Return tuple of lines (start-end point) that are used to plot gripper
        attached to the flange frame."""
        gripper_opening = self.gripper_opening / 2.0
        return (
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([0, +gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, -gripper_opening])).translation,
                (flange * SE2([self.gripper_length, -gripper_opening])).translation,
            ),
            (
                (flange * SE2([0, +gripper_opening])).translation,
                (flange * SE2([self.gripper_length, +gripper_opening])).translation,
            ),
        )

    def jacobian(self) -> np.ndarray:
        """Computes jacobian of the manipulator for the given structure and
        configuration."""
        #jac = np.zeros((3, len(self.q)))
        # todo: HW04 implement jacobian computation
        fi = [self.base_pose.rotation.angle]
        for i in range(self.dof):
            if self.structure[i] == 'R':    #R revolute
                fi.append(fi[i]+self.q[i])
            else:   #P prismatic
                fi.append(fi[i]+self.link_parameters[i])
        fi = fi[1:]

        x = []
        for i in range(self.dof):
            if self.structure[i] == 'R':    #R revolute
                n = 0
                for j in range(i,self.dof):
                    if self.structure[j] == 'R': 
                        n += -np.sin(fi[j])*self.link_parameters[j]
                    else:
                        n += -np.sin(fi[j])*self.q[j]
            else:   #P prismatic
                n = np.cos(fi[i])
            x.append(n)

        y = []
        for i in range(self.dof):
            if self.structure[i] == 'R':    #R revolute
                n = 0
                for j in range(i,self.dof):
                    if self.structure[j] == 'R': 
                        n += np.cos(fi[j])*self.link_parameters[j]
                    else:
                        n += np.cos(fi[j])*self.q[j]
            else:   #P prismatic
                n = np.sin(fi[i])
            y.append(n)

        theta = []
        for i in range(self.dof):
            if self.structure[i] == 'R':    #R revolute
                n = 1
            else:   #P prismatic
                n = 0
            theta.append(n)
        jac = np.array([x,y,theta])
        return jac

    def jacobian_finite_difference(self, delta=1e-5) -> np.ndarray:
        jac = np.zeros((3, len(self.q)))
        # todo: HW04 implement jacobian computation





        d = PlanarManipulator(link_parameters=self.link_parameters,
                              structure= self.structure,
                              base_pose= self.base_pose,
                              gripper_length=self.gripper_length)

        old = self.flange_pose()
        old = np.hstack((old.translation , old.rotation.angle))
        jac = np.array([[],[],[]])
        for i in range(self.dof):

            new_q = copy(self.q) 
            
            new_q[i] = new_q[i] + delta

            d.set_configuration(new_q)
            new = d.flange_pose()
            new = np.hstack([new.translation , new.rotation.angle])

            s = (new-old)/delta
            s = s.reshape(-1,1)
            
            jac = np.hstack((jac,s))


        return jac

    def ik_numerical(
        self,
        flange_pose_desired: SE2,
        max_iterations=1000,
        acceptable_err=1e-4,
    ) -> bool:
        """Compute IK numerically. Value self.q is used as an initial guess and updated
        to solution of IK. Returns True if converged, False otherwise."""
        # todo: HW05 implement numerical IK
        for i in range(max_iterations):
            c = self.flange_pose()
            e = np.append(flange_pose_desired.translation - c.translation , [flange_pose_desired.rotation.angle - c.rotation.angle])

            if np.linalg.norm(e) < acceptable_err:
                return True

            self.q = self.q + np.linalg.pinv(self.jacobian())@e
        return False

    def ik_analytical(self, flange_pose_desired: SE2) -> list[np.ndarray]:
        """Compute IK analytically, return all solutions for joint limits being
        from -pi to pi for revolute joints -inf to inf for prismatic joints."""
        assert self.structure in (
            "RRR",
            "PRR",
        ), "Only RRR or PRR structure is supported"

        ans = []
        # todo: HW05 implement analytical IK for RRR manipulator
        # todo: HW05 optional implement analytical IK for PRR manipulator
        if self.structure == "RRR":
            link = [self.link_parameters[2] ,0]
            s = flange_pose_desired.translation - flange_pose_desired.rotation.act(link)
            dxy = (s- self.base_pose.translation)
            d = np.linalg.norm(dxy)
            r1 = self.link_parameters[0]
            r2 = self.link_parameters[1]
            if (d > r1 + r2) or d < abs(r1-r2):
                return []
            
            a = (r1**2 - r2**2 + d**2)/(2*d)
            h = np.sqrt(r1**2-a**2)

            p = self.base_pose.translation + a * (dxy)/d
            rp = np.array([- dxy[1],dxy[0]]) * (h/d)
            if h==0:
                bod =[p]
            else:
                bod = [p+rp,p-rp]

            for b in bod:
                v1 = b-self.base_pose.translation
                v2 = s - b
                v3 = flange_pose_desired.translation - s
                new_q = [np.arctan2(v1[1],v1[0]) , np.arctan2(v2[1],v2[0]), np.arctan2(v3[1],v3[0])]

                for i in [2,1]:
                    new_q[i] = new_q[i]- new_q[i-1]
                new_q[0] = new_q[0] - self.base_pose.rotation.angle
                for i in range(3):
                    if new_q[i] > np.pi:
                        new_q[i] -= 2*np.pi
                    elif new_q[i] < -np.pi:
                        new_q[i] += 2*np.pi
                ans.append(new_q)

        elif self.structure == "PRR":
            link = [self.link_parameters[2] ,0]
            s = flange_pose_desired.translation - flange_pose_desired.rotation.act(link)

            r = self.link_parameters[1]
            x = self.base_pose.translation
            rot = SO2(angle= self.link_parameters[0])
            dx = rot.act(self.base_pose.rotation.act([1,0]))

            A = dx@dx
            B = 2 * ((x - s) @ dx)
            C = (x - s) @ (x - s) - r**2

            D = B**2 - 4 * A * C
            if D < 0:
                return[]
            elif np.isclose(D,0):
                t = [-B/(2*A)]
                
            elif D > 0:
                sqD = np.sqrt(D)
                t = [(-B + sqD)/(2*A), (-B - sqD)/(2*A)]

            for k in t:
                b = x + dx * k
                v1 = b-x
                v2 = s - b
                v3 = flange_pose_desired.translation - s
                new_q = [k , np.arctan2(v2[1],v2[0]), np.arctan2(v3[1],v3[0])]

                new_q[2] = new_q[2]- new_q[1]
                new_q[1] = new_q[1] - (self.base_pose.rotation.angle + self.link_parameters[0])
                for i in [1,2]:
                    if new_q[i] > np.pi:
                        new_q[i] -= 2*np.pi
                    elif new_q[i] < -np.pi:
                        new_q[i] += 2*np.pi
                ans.append(new_q)
        return ans

    def in_collision(self) -> bool:
        """Check if robot in its current pose is in collision."""
        frames = self.fk_all_links()
        points = [f.translation for f in frames]
        gripper_lines = self._gripper_lines(frames[-1])

        links = [LineString([a, b]) for a, b in zip(points[:-2], points[1:-1])]
        links += [MultiLineString((*gripper_lines, (points[-2], points[-1])))]
        for i in range(len(links)):
            for j in range(i + 2, len(links)):
                if links[i].intersects(links[j]):
                    return True
        return MultiLineString(
            (*gripper_lines, *zip(points[:-1], points[1:]))
        ).intersects(self.obstacles)
