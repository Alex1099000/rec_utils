import torch
import trimesh as tm
from pathlib import Path
from typing import Union, Optional
from rec_utils.structures import Scene, Frame
import numpy as np
from rec_utils.structures.utils import adjust_intrinsics
import cv2
from lietorch import SE3
from scipy.spatial.transform import Rotation as R

# def load_dust3r_scene(scene_dir):

#     output_params = torch.load(scene_dir / 'output_params.pt') if (scene_dir / 'output_params.pt').exists() else None
#     scene_params = torch.load(scene_dir / 'scene_params.pt')

#     return output_params, scene_params

def to_intrinsics_matrix(intrinsics):

    N = intrinsics.shape[0]
    K = np.zeros((N, 3, 3))
    
    K[:, 0, 0] = intrinsics[:, 0]  # fx
    K[:, 1, 1] = intrinsics[:, 1]  # fy
    K[:, 0, 2] = intrinsics[:, 2]  # cx
    K[:, 1, 2] = intrinsics[:, 3]  # cy
    K[:, 2, 2] = 1.0               
    
    return K

def to_se3_matrix(pvec):
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(pvec[3:]).as_matrix()
    pose[:3, 3] = pvec[:3]
    # return np.linalg.inv(pose)
    return pose

def inverse_depth_to_depth(inverse_depths, eps = 1e-1):

    valid_mask = (inverse_depths > eps) 
    depths = np.full_like(inverse_depths, 0.0)
    depths[valid_mask] = 1.0 / inverse_depths[valid_mask]
    
    return depths

def load_droid_scene(scene_dir):

    images = np.load(scene_dir / 'images.npy')
    images = images[:, [2, 1, 0], :, :].transpose(0, 2, 3, 1)

    depths = np.load(scene_dir / 'disps.npy')
    depths = inverse_depth_to_depth(depths)
    
    poses = np.load(scene_dir / 'poses.npy')
    # poses = SE3(torch.from_numpy(poses).float()).matrix().numpy() # SE3(poses).inv().matrix()
    poses = np.array([to_se3_matrix(pose) for pose in poses])
    
    intrinsics = np.load(scene_dir / 'intrinsics.npy')
    intrinsics = to_intrinsics_matrix(intrinsics)
    
    frame_ids = np.load(scene_dir / 'tstamps.npy')

    return images, depths, poses, intrinsics, frame_ids

class DROIDDataset:
    def __init__(self, root_dir: Path, scenes: Union[list[str], Optional[None]]=None):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir
        self.scene_ids = scenes if scenes is not None else [a.stem for a in root_dir.iterdir() if a.is_dir()]
        self.scenes = [None for _ in range(len(self.scene_ids))]

        self.scene_id2index = {scene_id: index for index, scene_id in enumerate(self.scene_ids)}

    def load_scene(self, index):
        if self.scenes[index] is not None:
            return self.scenes[index]
        self.scenes[index] = DROIDScene(self.root_dir / self.scene_ids[index])
        return self.scenes[index]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.scene_id2index[index]
        if isinstance(index, int):
            index = index
            return self.load_scene(index)

        raise ValueError(f"Invalid index type {type(index)}")


    def __repr__(self):
        return f"DROIDDataset(root_dir={self.root_dir}, num_scenes={len(self.scenes)})"

class DROIDScene(Scene):
    def __init__(self, root_dir: Path):
        super().__init__(root_dir)

    def get_frame_list(self):
        self.frames = []
        # output_params, scene_params = load_dust3r_scene(self.root_dir)
        images, depths, poses, intrinsics, frame_ids = load_droid_scene(self.root_dir)

        # poses = scene_params['poses']
        # pts3d = scene_params['pts3d']
        # Ks = scene_params['Ks']
        # image_names = scene_params['image_files']
        # confs = scene_params['im_conf']
        # depths = scene_params['depths']
        
        for i in range(len(frame_ids)):
            frame_id = frame_ids[i]
            image = images[i]
            pose = poses[i]
            intrinsic = intrinsics[i]
            depth = depths[i]
            confidence = 1.0
            # output_params = output_params

            self.frames.append(DROIDFrame(frame_id=frame_id, image=image, pose=pose, intrinsics=intrinsic, depth=depth, confidence=confidence))

        return self.frames


class DROIDFrame(Frame):
    def __init__(self, frame_id, image, pose, intrinsics, depth, confidence):
        # pose = pose.numpy()
        # intrinsics = intrinsics.numpy()
        # depth = depth.numpy()
        # confidence = confidence.numpy()

        super().__init__(pose=pose, depth_intrinsics=intrinsics)
        self.confidence = confidence
        # self.output_params = output_params
        self._depth = depth
        self._image = image
        self._frame_id = frame_id

    @property
    def depth(self):
        # if self._frame_id < 2:
        #     return None
        # else:
        return self._depth
    
    @property
    def image(self):
        return self._image

    # @property
    # def frame_id(self):
    #     return Path(self.image_path).stem
    @property
    def frame_id(self):
        return str(int(self._frame_id)).zfill(6)

    def get_pcd(self, confidence_threshold=0, depth_truncation=4.0):
        pose = np.eye(4)
        if self.pose is not None:
            pose = self.pose
            
        
        if self._image_intrinsics is None and self._depth_intrinsics is None:
            raise ValueError("Image and depth intrinsics are both None")
        image = self.image
        depth = self.depth

        if depth is None:
            raise ValueError("Depth is None")
        if image is None:
            image = np.zeros((depth.shape[0], depth.shape[1], 3))

        intrinsics = self.unscaled_intrinsics
        depth_intrinsics = adjust_intrinsics(intrinsics, (1, 1), self.depth_shape)
        depth_shape = self.depth_shape
        image = cv2.resize(image, (depth_shape[1], depth_shape[0]), interpolation=cv2.INTER_LINEAR)

        y, x = np.where((depth > 0) & (self.confidence > confidence_threshold) & (depth < depth_truncation))
        colors = image[y, x]
        pixel_coords = np.vstack([x, y, np.ones(len(x))])
        normalized_coords = np.linalg.inv(depth_intrinsics)[:3, :3] @ pixel_coords

        z = depth[y, x]

        camera_points_3d = normalized_coords * z
        camera_points_homogeneous = np.vstack([camera_points_3d, np.ones(len(x))])
        world_points_homogeneous = pose @ camera_points_homogeneous

        points = world_points_homogeneous[:3, :] / world_points_homogeneous[3, :]
        points = points.T

        return points, colors

# /home/jovyan/users/kolodiazhnyi/zakharov/Indoor/SLAM/DROID-SLAM/n_1500_npy/75d29d69b8
