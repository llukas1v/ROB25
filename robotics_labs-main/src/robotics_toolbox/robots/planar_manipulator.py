#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved 
# Created on: 2023-08-21
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing planar manipulator."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from shapely import MultiPolygon, LineString, MultiLineString

from robotics_toolbox.core import SE2, SE3
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
        #OK todo HW02: implement fk for the flange
        act = self.base_pose
        for i in range(len(self.structure)):

            l_i = self.link_parameters[i]
            q_i = self.q[i]

            if self.structure[i] == "R":
                T_tmp = SE2(rotation = q_i)* SE2(translation = [l_i,0]) #nejprve otoceni a potom posun o pevnou delku
            elif self.structure[i] == "P":
                T_tmp = SE2(rotation = l_i)* SE2(translation = [q_i,0]) #nejprve zadani pevneho natoceni a pak posun
            else:
                raise ValueError(f"Unknown joint type (just R or P is possible): {self.structure[i]}")
            act = act*T_tmp
            
        return act

    def fk_all_links(self) -> list[SE2]:
        """Compute FK for frames that are attached to the links of the robot.
        The first frame is base_frame, the next frames are described in the constructor.
        """

        #OK todo HW02: implement fk
        frames = [self.base_pose]
        for i in range(len(self.structure)):
            act = frames[i]
            l_i = self.link_parameters[i]
            q_i = self.q[i]
            if self.structure[i] == "R":
                T_tmp = SE2(rotation = q_i)* SE2(translation = [l_i,0]) #nejprve otoceni a potom posun o pevnou delku
            elif self.structure[i] == "P":
                T_tmp = SE2(rotation = l_i)* SE2(translation = [q_i,0]) #nejprve zadani pevneho natoceni a pak posun
            else:
                raise ValueError(f"Unknown joint type (just R or P is possible): {self.structure[i]}")
            act = act*T_tmp
            frames.append(act)
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
        n = len(self.q)
        jac = np.zeros((3, len(self.q)))
        # OK: HW04 implement jacobian computation

        #calculation of theta star
        theta_star = []
        for j in range(n):
            theta_j_star = self.base_pose.rotation.angle  #base position
            for k in range(j+1):                          #sum of all previous angles pro R is q_k for P is link parameter
                if self.structure[k] == "R":
                    theta_j_star += self.q[k]
                elif self.structure[k] == "P":
                    theta_j_star += self.link_parameters[k]
                else:
                    raise ValueError(f"Unknown joint type (just R or P is possible): {self.structure[j]}")
            theta_star.append(theta_j_star)

        #calculation of jacobian
        for i in range(n):
            #for prismatic joint we need to calculate theta star up to i plus link parameter
            if self.structure[i] == "P":
                if i == 0:
                    theta_star_prismatic = self.base_pose.rotation.angle + self.link_parameters[i]
                else:
                    theta_star_prismatic = theta_star[i-1] + self.link_parameters[i]

                #assign values to jacobian matrix like expleined in the lecture
                jac[0,i] = np.cos(theta_star_prismatic)
                jac[1,i] = np.sin(theta_star_prismatic)
                jac[2,i] = 0.0

            #for revolute joint we need to sum all connection from i to n
            elif self.structure[i] =="R":
                sum_dx = 0.0
                sum_dy = 0.0
                for j in range(i,n): #make sum from i to n, for R use link_parametr, for P use q_j
                    if self.structure[j] == "R": 
                        l_j = self.link_parameters[j]
                    elif self.structure[j] == "P":
                        l_j = self.q[j]
                    else:
                        raise ValueError(f"Unknown joint type (just R or P is possible): {self.structure[j]}")
                    sum_dx += -l_j * np.sin(theta_star[j])
                    sum_dy += l_j * np.cos(theta_star[j])
                
                #assign values to jacobian matrix like expleined in the lecture
                jac[0,i] = sum_dx
                jac[1,i] = sum_dy
                jac[2,i] = 1.0
            else:
                raise ValueError(f"Unknown joint type (just R or P is possible): {self.structure[j]}")

        return jac

    def jacobian_finite_difference(self, delta=1e-5) -> np.ndarray:
        jac = np.zeros((3, len(self.q)))
        # OK: HW04 implement jacobian computation
        flangle_original = self.flange_pose() #original flangle position
        for i in range(len(self.q)):
            q_default = self.q[i]      
            self.q[i] = q_default + delta         #move to another position
            flangle_delta = self.flange_pose()    #flangle position after move for delta
            
            #computation of translatiom difference, form definition of difference. 
            flangle_difference_tr = flangle_delta.translation - flangle_original.translation
            diff_tr = flangle_difference_tr/delta

            #computation of theta difference, form definition of difference. 
            flangle_difference_theta = flangle_delta.rotation.angle - flangle_original.rotation.angle
            diff_theta = flangle_difference_theta/delta

            #update jacobian
            jac[0,i] = diff_tr[0]
            jac[1,i] = diff_tr[1]
            jac[2,i] = diff_theta

            #set original configuration
            self.q[i] = q_default
        return jac

    def ik_numerical(
        self,
        flange_pose_desired: SE2,
        max_iterations=1000,
        acceptable_err=1e-4,
    ) -> bool:
        """Compute IK numerically. Value self.q is used as an initial guess and updated
        to solution of IK. Returns True if converged, False otherwise."""
         #OK todo: HW05 implement numerical IK
        for i in range(max_iterations):
            #ziskani akutalni pozice
            act_pos = self.flange_pose()

            #vypocet chyby pro SE2 nexistuje metoda pro odecteni, musi se tedy postupovat po slozkach
            #translation error
            err_pos = act_pos.translation - flange_pose_desired.translation
            #rotation error a prevedeni do intervalu -pi az pi
            err_rot = act_pos.rotation.angle - flange_pose_desired.rotation.angle   
            err_rot = np.arctan2(np.sin(err_rot), np.cos(err_rot))
            #normalizace error
            err =   np.array([err_pos[0],err_pos[1],err_rot]) 

        
            #kontrola, zda je jiz splnena podminaka 
            if np.linalg.norm(err) < acceptable_err:
                return True
            
            #vypocet jakobianu a jeho pseudoinverze dle emailu
            J = self.jacobian()
            J_inv = np.linalg.pinv(J)

            #aktualizace konfigurace
            self.q = self.q - (J_inv @ err)
        return False

    def ik_analytical(self, flange_pose_desired: SE2) -> list[np.ndarray]:
        """Compute IK analytically, return all solutions for joint limits being
        from -pi to pi for revolute joints -inf to inf for prismatic joints."""
        assert self.structure in (
            "RRR",
            "PRR",
        ), "Only RRR or PRR structure is supported"
        #OK todo: HW05 implement analytical IK for RRR manipulator
        #import gerometrickych nastoroju
        import robotics_toolbox.utils.geometry_utils as utils
        solutions = []

        if self.structure == "RRR":
            #ziskani koncove pozice a oreintace ze zadani 
            xd = flange_pose_desired.translation[0]
            yd = flange_pose_desired.translation[1]
            thetad = flange_pose_desired.rotation.angle


            #ziskani delky jednotlivych clanku 
            l1 = self.link_parameters[0]
            l2 = self.link_parameters[1]
            l3 = self.link_parameters[2]
            
            #zsikani pozice zaklandny
            base = self.base_pose.translation
            q0 = self.base_pose.rotation.angle

            #vypocet pozice 3. kloubu
            x_3 = xd - l3 * np.cos(thetad)
            y_3 = yd - l3 * np.sin(thetad)
            
            #vypocet vsech moznich pruseciku kruznic 
            intersection_points = utils.circle_circle_intersection(
                c0 = base, r0 = l1,
                c1 = [x_3, y_3], r1 = l2
            )

            #uceni uhlu pro jednotlive reseni
            for intersection in intersection_points:
                x_int = intersection[0]
                y_int = intersection[1]

                #vypocet prvniho kloubu, vypocutu uhel mezi zaklanou a prusecikem, pote odecist uhel base
                q1 = np.arctan2(y_int - base[1] , x_int - base[0]) - q0
                
                #vypocet druheho kloubu, vypocet uhlu mezi prusecikem a pozici 3. kloubu, pote odecistu prvni a base uhel
                q2 = np.arctan2(y_3 - y_int, x_3 - x_int) - q1 - q0
                
                #vypocet tetiho kloubu jednoduse odectenim vesch predchozich uhlu od pozadovane orietnace
                q3 = thetad - q1 - q2 -q0
                
                #normalizace jednotlivych uhlu do intervali -pi az pi
                q1 = np.arctan2(np.sin(q1), np.cos(q1))  
                q2 = np.arctan2(np.sin(q2), np.cos(q2))
                q3 = np.arctan2(np.sin(q3), np.cos(q3))

                #pripojeni reseni do pole
                solutions.append( np.array([q1, q2, q3]) )
               
        
        # todo: HW05 optional implement analytical IK for PRR manipulator
        if self.structure == "PRR":
            #ziskani koncove pozice a oreintace ze zadani 
            xd = flange_pose_desired.translation[0]
            yd = flange_pose_desired.translation[1]
            thetad = flange_pose_desired.rotation.angle


            #ziskani uhlu v pripade 0 a delky jednotlivych clanku v pripade 1 a 2 
            alpha = self.link_parameters[0]
            l2 = self.link_parameters[1]
            l3 = self.link_parameters[2]

            #zsikani pozice zaklandny
            base = self.base_pose.translation
            q0 = self.base_pose.rotation.angle

            #vypocet pozice 2. kloubu
            x_2 = xd - l3 * np.cos(thetad)
            y_2 = yd - l3 * np.sin(thetad)

            #urceni druheho bodu na primce
            point2x = base[0] + 10*np.cos(q0+alpha)
            point2y = base[1] + 10*np.sin(q0+alpha)

            #ziskani jednitlivych pruseciku
            intersection_points = utils.circle_line_intersection(
                c = (x_2,y_2), r = l2,
                a = base, b = (point2x,point2y) 
            )

            #vyhodnoceni jednolivych pruseciku
            for intersection in intersection_points:
                x_int = intersection[0]
                y_int = intersection[1]

                #vypocet vzdalenosti na ose posunu, tedy ose x. Zjdnoduseni projekce na primku. 
                q1 = np.dot(
                            np.array([x_int - base[0], y_int - base[1]]),
                            np.array([np.cos(q0+alpha), np.sin(q0+alpha)])
                        )
                
                #vypocet prvniho kloubu, vypocet uhlu mezi prusecikem a pozici 2. kloubu, pote odecistu alpha a base uhel
                q2 = np.arctan2(y_2 - y_int, x_2 - x_int) - alpha - q0
                
                #vypocet druheho kloubu jednoduse odectenim vesch predchozich uhlu od pozadovane orietnace
                q3 = thetad - q0 - alpha - q2
                
                #normalizace jednotlivych uhlu do intervali -pi az pi
                q2 = np.arctan2(np.sin(q2), np.cos(q2))
                q3 = np.arctan2(np.sin(q3), np.cos(q3))

                #pripojeni reseni do pole
                solutions.append( np.array([q1, q2, q3]) )

        return solutions

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
