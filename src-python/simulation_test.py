import sapien
import numpy as np
from PIL import Image

def main():
    _scene : sapien.Scene = sapien.Scene()
    _scene.set_timestep(1 /100.0) # TODO that should be a parameter

    # Set some default params for light etc.
    _scene.add_ground(0)
    _scene.set_ambient_light([0.5, 0.5, 0.5])
    _scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    # TODO: Now the scene is hardcoded change to some scene description file
    asset_path = "/home/nick/git/khafre/assets/"
    loader = _scene.create_urdf_loader()
    loader.fix_root_link = False
    _table = loader.load(asset_path+"table/table.urdf")
    _table.set_root_pose(sapien.Pose([0, 0, 0.44], [1, 0, 0, 0]))

    print(_table)

    builder = _scene.create_actor_builder()
    builder.add_convex_collision_from_file(
        filename=asset_path+"beermug/BeerMugCollision.obj"
    )
    builder.add_visual_from_file(
        filename=asset_path+"beermug/BeerMugVisual.obj"
    )
    _mug = builder.build(name="mug")
    _mug.set_pose(sapien.Pose(p=[-0.2, 0, 0.44 + 0.05]))

    near, far = 0.1, 100
    width, height = int(640/2), int(480/2)

    # Compute the camera pose by specifying forward(x), left(y) and up(z)
    cam_pos = np.array([-2, -2, 3])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos

    camera = _scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(35),
        near=near,
        far=far,
    )
    camera.entity.set_pose(sapien.Pose(mat44))

    #TODO Viewer is mainly for debug. Should be instantiated with a param.
    _viewer = _scene.create_viewer()  # Create a viewer (window)
    # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
    # The principle axis of the camera is the x-axis
    _viewer.set_camera_xyz(x=-4, y=0, z=2)

    # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # The camera now looks at the origin
    _viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
    _viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
    while not _viewer.closed:  # Press key q to quit
        _scene.step()  # Simulate the world
        _scene.update_render()  # Update the world to the renderer
        camera.take_picture()  # submit rendering jobs to the GPU
        rgba = camera.get_picture("Color")  # [H, W, 4]
        print(rgba)
        print(type(rgba))
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        rgba_pil = Image.fromarray(rgba_img)
        rgba_pil.save("/tmp/color.png")
        _viewer.render()

if __name__ == "__main__":
    main()
