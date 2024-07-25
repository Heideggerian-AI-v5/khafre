import sapien
import numpy as np

import cv2 as cv

from khafre.bricks import ReifiedProcess

class SapienSim(ReifiedProcess):
    """
    Subprocess that uses a simulatorto produce data.

    Wires supported by this subprocess:
    OutImg: publisher. Output image from a camera.
    """
    def __init__(self):
        super().__init__()
        
        self._rateMask = None
        self._droppedMask = 0
        self._rateDepth = None
        self._droppedDepth = 0
        
        """TODO: Initialize the simulator object and the scene"""
        self._scene : sapien.Scene = sapien.Scene()
        self._scene.set_timestep(1 /100.0) # TODO that should be a parameter

        # Set some default params for light etc.
        self._scene.add_ground(0)
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        self._scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        # TODO: Now the scene is hardcoded change to some scene description file
        asset_path = "/home/nick/git/khafre/assets/"
        loader = self._scene.create_urdf_loader()
        self._table = loader.load(asset_path+"table/table.urdf")
        self._table.set_root_pose(sapien.Pose([0, 0, 0.44], [1, 0, 0, 0]))

        builder = self._scene.create_actor_builder()
        builder.add_convex_collision_from_file(
            filename=asset_path+"beermug/BeerMugCollision.obj"
        )
        builder.add_visual_from_file(
            filename=asset_path+"beermug/BeerMugVisual.obj"
        )
        self._mug = builder.build(name="mug")
        self._mug.set_pose(sapien.Pose(p=[-0.2, 0, 0.44 + 0.05]))

        #TODO Viewer is mainly for debug. Should be instantiated with a param.
        # self._viewer = self._scene.create_viewer()  # Create a viewer (window)
        # # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
        # # The principle axis of the camera is the x-axis
        # self._viewer.set_camera_xyz(x=-4, y=0, z=2)

        # # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
        # # The camera now looks at the origin
        # self._viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
        # self._viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        # Add a camera to get images
        # TODO: remove hardcoded stuff
        near, far = 0.1, 100
        width, height = 640, 480

        # Compute the camera pose by specifying forward(x), left(y) and up(z)
        cam_pos = np.array([-2, -2, 3])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos

        self._camera = self._scene.add_camera(
            name="camera",
            width=width,
            height=height,
            fovy=np.deg2rad(35),
            near=near,
            far=far,
        )
        self._camera.entity.set_pose(sapien.Pose(mat44))
    
    def _checkPublisherRequest(self, name, queues, consumerSHM):
        return name in {"SimCam", "DbgSimCam"}
    
    def doWork(self):
        """TODO: here is the actual work"""
        print("WORKING")
        self._scene.step()
        self._scene.update_render()
        self._camera.take_picture()  # submit rendering jobs to the GPU
        
        # rgba is a numpy array
        # rgba = self._camera.get_picture("Color")  # [H, W, 4]
        
        # if "DbgSimCam" in self._publishers:
        #         # Here we can hog the shared memory as long as we like -- dbgvis won't use it until we notify it that there's a new frame to show.
        #         with self._publishers["DbgImg"] as dbgImg:
        #             workImg = dbgImg
        #             if (rgba.shape[0] != dbgImg.shape[0]) or (rgba.shape[1] != dbgImg.shape[1]):
        #                 np.copyto(dbgImg, cv.resize(workImg, (dbgImg.shape[1], dbgImg.shape[0]), interpolation=cv.INTER_LINEAR))
        #         self._publishers["DbgSimCam"].sendNotifications("%.02f %.02f ifps | %d%% %d%% obj drop" % (self._rateMask if self._rateMask is not None else 0.0, self._rateDepth if self._rateDepth is not None else 0.0, self._droppedMask, self._droppedDepth))

        """
        TODO The viewer doesn't seem to be working. 
        Something seems to be blocking, which usually shouldn't be the case.
        """
        # self._viewer.render()