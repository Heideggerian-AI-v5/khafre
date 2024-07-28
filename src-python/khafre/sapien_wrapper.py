import cv2 as cv
from multiprocessing import Queue
import numpy as np
import sapien

import time

from khafre.bricks import ReifiedProcess

class SapienSim(ReifiedProcess):
    """
    Subprocess that uses a simulator to produce data.

    Wires supported by this subprocess:
    OutImg: publisher. Output image from a camera.
    DbgImg: publisher. Output image for debug visualizer.
    """
    def __init__(self, dt=0.01,near=0.1,far=100,width=640,height=480, viewer=False):
        super().__init__()
        
        self._rateMask = None
        self._droppedMask = 0
        self._rateDepth = None
        self._droppedDepth = 0
        self._scene = None
        self._loader = None
        self._builder = None
        self._viewer = None
        self._haveViewer = viewer
        self._dt = dt
        self._near = near
        self._far = far
        self._width = width
        self._height = height
        self._camera = None
        self._command = Queue()
        self._assets = {}
        self._actors = {}
        
    def _checkPublisherRequest(self, name, queues, consumerSHM):
        return (name in {"OutImg", "DbgImg"}) and ((self._height, self._width)==tuple(consumerSHM._shape[:2]))
    def sendCommand(self, command, block=False, timeout=None):
        self._command.put(command, block=block, timeout=timeout)
    def onStart(self):
        self._assets = {}
        self._actors = {}
        self._scene : sapien.Scene = sapien.Scene()
        self._scene.set_timestep(self._dt) # TODO that should be a parameter
        self._scene.add_ground(0)
        self._loader = self._scene.create_urdf_loader()
        self._builder = self._scene.create_actor_builder()
        if self._haveViewer:
            self._viewer = self._scene.create_viewer()  # Create a viewer (window)
            # # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
            # # The principle axis of the camera is the x-axis
            self._viewer.set_camera_xyz(x=-4, y=0, z=2)
            # # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
            # # The camera now looks at the origin
            self._viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
            self._viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
        self._camera = self._scene.add_camera(
            name="camera",
            width=self._width,
            height=self._height,
            fovy=np.deg2rad(35),
            near=self._near,
            far=self._far,
        )
    def _handleCommand(self, command):
        op, args = command
        if "SET AMBIENT LIGHT" == op:
            color = args[0]
            self._scene.set_ambient_light(color)
        elif "ADD DIRECTIONAL LIGHT" == op:
            direction, color = args
            self._scene.add_directional_light(direction, color)
        elif "LOAD ASSET" == op:
            name, urdfPath, pos, orn = args
            self._assets[name] = self._loader.load(urdfPath)
            self._assets[name].set_root_pose(sapien.Pose(pos, orn))
        elif "LOAD ACTOR" == op:
            name, collisionModelPath, visualModelPath, pos, orn = args
            self._builder.add_convex_collision_from_file(filename=collisionModelPath)
            self._builder.add_visual_from_file(filename=visualModelPath)
            self._actors[name] = self._builder.build(name="mug")
            self._actors[name].set_pose(sapien.Pose(pos, orn))
        elif "SET CAMERA POSE" == op:
            """
            TODO: what if cross of cam_pos and up is 0? And how can we set pos and orn independently?
            """
            cam_pos = args[0]
            forward = -cam_pos / np.linalg.norm(cam_pos)
            left = np.cross([0, 0, 1], forward)
            left = left / np.linalg.norm(left)
            up = np.cross(forward, left)
            mat44 = np.eye(4)
            mat44[:3, :3] = np.stack([forward, left, up], axis=1)
            mat44[:3, 3] = cam_pos
            self._camera.entity.set_pose(sapien.Pose(mat44))
        else:
            pass
    def doWork(self):
        """TODO: move command interface to ReifiedProcess
        """
        while not self._command.empty():
            self._handleCommand(self._command.get())
        """TODO: here is the actual work"""
        self._scene.step()
        self._scene.update_render()
        self._camera.take_picture()
        
        # rgba is a numpy array
        rgb = self._camera.get_picture("Color")[:, :, [2,1,0]]
        outImg = (rgb * 255).clip(0, 255).astype(np.uint8)
        if "DbgImg" in self._publishers:
            self._publishers["DbgImg"].publish(rgb, "")
        if "OutImg" in self._publishers:
            self._publishers["OutImg"].publish(outImg, {"imgId": str(time.perf_counter())})
        if self._haveViewer:
            self._viewer.render()
