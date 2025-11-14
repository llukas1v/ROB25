#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 3D rotation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class SO3:
    """This class represents an SO3 rotations internally represented by rotation
    matrix."""

    def __init__(self, rotation_matrix: ArrayLike | None = None) -> None:
        """Creates a rotation transformation from rot_vector."""
        super().__init__()
        self.rot: np.ndarray = (
            np.asarray(rotation_matrix) if rotation_matrix is not None else np.eye(3)
        )

    @staticmethod
    def exp(rot_vector: ArrayLike) -> SO3:
        """Compute SO3 transformation from a given rotation vector, i.e. exponential
        representation of the rotation."""
        v = np.asarray(rot_vector)
        assert v.shape == (3,)
        t = SO3()
        #OK todo HW01: implement Rodrigues' formula, t.rot = ...
        theta = np.linalg.norm(v)
        I = np.eye(3)
        if np.isclose(theta, 0):
            t.rot = I
            return t
        v= v/theta
        Skew = np.array([[0,-v[2],v[1]],
                         [v[2],0, -v[0]],
                         [-v[1],v[0],0]])
        t.rot = I + np.sin(theta)*Skew + (1-np.cos(theta))*(Skew@Skew)
        return t

    def log(self) -> np.ndarray:
        """Compute rotation vector from this SO3"""
        #OK todo HW01: implement computation of rotation vector from this SO3
        R = self.rot
        theta = np.arccos((np.trace(R) - 1) / 2)
        if np.isclose(theta, 0):
            return np.zeros(3)
        Skew = (R - R.T)/(2*np.sin(theta))
        v = theta *np.array([Skew[2,1],Skew[0,2],Skew[1,0]])
        return v

    def __mul__(self, other: SO3) -> SO3:
        """Compose two rotations, i.e., self * other"""
        #OK todo: HW01: implement composition of two rotation.
        Comp = SO3()
        Comp.rot = self.rot @ other.rot
        return Comp

    def inverse(self) -> SO3:
        """Return inverse of the transformation."""
        #OK todo: HW01: implement inverse, do not use np.linalg.inverse()
        inv = SO3()
        inv.rot = self.rot.T
        return inv

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given vector by this transformation."""
        v = np.asarray(vector)
        assert v.shape == (3,)
        return self.rot @ v

    def __eq__(self, other: SO3) -> bool:
        """Returns true if two transformations are almost equal."""
        return np.allclose(self.rot, other.rot)

    @staticmethod
    def rx(angle: float) -> SO3:
        """Return rotation matrix around x axis."""
        #OK todo: HW1opt: implement rx
        C = np.cos(angle)
        S = np.sin(angle)
        RX = SO3()
        RX.rot = np.array([[1,0,0],
                           [0,C,-S],
                           [0,S,C]])
        return RX
        #raise NotImplementedError("RX needs to be implemented.")

    @staticmethod
    def ry(angle: float) -> SO3:
        """Return rotation matrix around y axis."""
        #OK todo: HW1opt: implement ry
        C = np.cos(angle)
        S = np.sin(angle)
        RY = SO3()
        RY.rot = np.array([[C,0,S],
                           [0,1,0],
                           [-S,0,C]])
        return RY
        #raise NotImplementedError("RY needs to be implemented.")

    @staticmethod
    def rz(angle: float) -> SO3:
        """Return rotation matrix around z axis."""
        C = np.cos(angle)
        S = np.sin(angle)
        RZ = SO3()
        RZ.rot = np.array([[C,-S,0],
                           [S,C,0],
                           [0,0,1]])
        return RZ
        #OK todo: HW1opt: implement rz
        #raise NotImplementedError("RZ needs to be implemented.")

    @staticmethod
    def from_quaternion(q: ArrayLike) -> SO3:
        """Compute rotation from quaternion in a form [qx, qy, qz, qw]."""
        #OK todo: HW1opt: implement from quaternion
        R = SO3()
        q_xyz= q[:3]

        theta = 2*np.arccos(q[3])
        v = q_xyz / np.sin(theta/2)
        Skew = np.array([[0,-v[2],v[1]],
                         [v[2],0, -v[0]],
                         [-v[1],v[0],0]])
        I = np.eye(3)
        R.rot = I+ np.sin(theta)*Skew + (1-np.cos(theta))*(Skew@Skew)
        return R

    def to_quaternion(self) -> np.ndarray:
        """Compute quaternion from self."""
        R = self.rot
        q = np.zeros(4)
        qw = np.sqrt(1.0 + np.trace(R)) / 2.0
        q[3] = qw
        q[0] = (R[2,1] - R[1,2]) / (4*qw)
        q[1] = (R[0,2] - R[2,0]) / (4*qw)
        q[2] = (R[1,0] - R[0,1]) / (4*qw)
        q = q / np.linalg.norm(q)  #This vector needs to be normalized by definition
        return q
        

        #OK todo: HW1opt: implement to quaternion
        #raise NotImplementedError("To quaternion needs to be implemented")

    @staticmethod
    def from_angle_axis(angle: float, axis: ArrayLike) -> SO3:
        """Compute rotation from angle axis representation."""
        #OK todo: HW1opt: implement from angle axis
        R = SO3()
        theta = angle
        v = np.asarray(axis)
        v = v / np.linalg.norm(v)         #This vector needs to be normalized by definition
        Skew = np.array([[0,-v[2],v[1]],
                         [v[2],0, -v[0]],
                         [-v[1],v[0],0]])
        I = np.eye(3)
        R.rot = I+ np.sin(theta)*Skew + (1-np.cos(theta))*(Skew@Skew)
        return R
        
        #raise NotImplementedError("Needs to be implemented")

    def to_angle_axis(self) -> tuple[float, np.ndarray]:
        """Compute angle axis representation from self."""
        #OK todo: HW1opt: implement to angle axis
        R = self.rot
        I = np.eye(3)
        axis = np.zeros(3)
        theta = np.arccos((np.trace(R) - 1) / 2)
        if np.allclose(R, I):
            return 0, axis

        

        elif np.isclose(theta, np.pi):
            if not np.isclose(1 + R[2, 2], 0):
                axis = np.array([R[0, 2], R[1, 2], 1 + R[2, 2]])
                axis = axis/np.sqrt(2*(1 + R[2, 2]))
            elif not np.isclose(1 + R[1, 1], 0):
                axis = np.array([R[0, 2], 1 + R[1, 1], R[2, 1]])
                axis = axis/np.sqrt(2*(1 + R[1, 1]))
            else:
                axis = np.array([1 + R[0, 0], R[1, 0], R[2, 0]])
                axis = axis/np.sqrt(2*(1 + R[0, 0]))

        else:

            Skew = (R - R.T)/(2*np.sin(theta))
            axis = theta *np.array([Skew[2,1],Skew[0,2],Skew[1,0]])
        
        axis = axis / np.linalg.norm(axis)
        return theta, axis
        
        #raise NotImplementedError("Needs to be implemented")

    @staticmethod
    def from_euler_angles(angles: ArrayLike, seq: list[str]) -> SO3:
        """Compute rotation from euler angles defined by a given sequence.
        angles: is a three-dimensional array of angles
        seq: is a list of axis around which angles rotate, e.g. 'xyz', 'xzx', etc.
        """
        #OK todo: HW1opt: implement from euler angles
        #for we seqventlz use rotation matrix around axis
        R = SO3()
        def axis_rot (angle: float,axis: str):
            S = np.sin(angle)
            C = np.cos(angle)
            if (axis == "x"):
                return np.array([[1,0,0],
                      [0,C,-S],
                      [0,S,C]])
            elif (axis == "y"):
                return np.array([[C,0,S],
                      [0,1,0],
                      [-S,0,C]])
            else: #axis z
                return np.array([[C,-S,0],
                      [S,C,0],
                      [0,0,1]])
        R.rot = axis_rot(angles[0],seq[0]) @ axis_rot(angles[1],seq[1]) @ axis_rot(angles[2],seq[2])
        return R
        #raise NotImplementedError("Needs to be implemented")

    def __hash__(self):
        return id(self)
