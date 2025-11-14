#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 2D transformation."""

from __future__ import annotations

import copy
import numpy as np
from numpy.typing import ArrayLike
from robotics_toolbox.core import SO2


class SE2:
    """Transformation in 2D that is composed of rotation and translation."""

    def __init__(
        self, translation: ArrayLike | None = None, rotation: SO2 | float | None = None
    ) -> None:
        """Crete an SE2 transformation. Identity is the default. The se3 instance
        is represented by translation and rotation, where rotation is SO2 instance.

        Attributes:
        :param translation: 2D vector representing translation
        :param rotation: SO2 rotation or angle in radians that is converted to SO2
        """
        super().__init__()
        self.translation = (
            np.asarray(translation) if translation is not None else np.zeros(2)
        )
        assert self.translation.shape == (2,)
        if isinstance(rotation, SO2):
            self.rotation = rotation
        elif isinstance(rotation, float):
            self.rotation = SO2(rotation)
        else:
            self.rotation = SO2()

    def __mul__(self, other: SE2) -> SE2:
        """Compose two transformation, i.e., self * other"""
        #OK todo: HW01: implement composition of two transformation.\
        #we need homogenus matrix
        H_12 = self.homogeneous() @ other.homogeneous()
        #now we make homogenous matrix of composition so we neet to separte translation and rotation
        Comp = SE2()
        #Comp.homogeneous= T_12
        Comp.translation = H_12[:2, 2]
        Comp.rotation = SO2() #for rotation matrix we need SO2 matrix
        Comp.rotation.rot =  H_12[:2, :2] 
        return  Comp

    def inverse(self) -> SE2:
        """Compute inverse of the transformation. Do not use np.linalg.inv."""
        #OK todo: HW1 implement inverse
        R_T = self.rotation.rot.T 
        inv = SE2()
        inv.rotation = SO2()
        inv.rotation.rot = R_T
        inv.translation = -R_T @ self.translation
        return inv

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Transform given 2D vector by this SE2 transformation."""
        v = np.asarray(vector)
        assert v.shape == (2,)
        # todo: HW1 implement transformation of a given vector
        v = self.rotation.rot @ vector + self.translation
        return v

    def set_from(self, other: SE2):
        """Copy the properties into current instance."""
        self.translation = copy.deepcopy(other.translation)
        self.rotation = copy.deepcopy(other.rotation)

    def homogeneous(self) -> np.ndarray:
        """Return homogeneous transformation matrix."""
        h = np.eye(3)
        h[:2, :2] = self.rotation.rot
        h[:2, 2] = self.translation
        return h

    def __eq__(self, other: SE2) -> bool:
        """Returns true if two transformations are almost equal."""
        return (
            np.allclose(self.translation, other.translation)
            and self.rotation == other.rotation
        )

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return (
            f"SE2(translation={self.translation}, rotation=SO2({self.rotation.angle}))"
        )
