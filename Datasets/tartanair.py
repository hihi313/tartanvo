import os
from pathlib import Path
import re
from typing import List

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .transformation import SEs2ses, pos_quats2SEs, pose2motion
from .utils import make_intrinsics_layer


class CameraIntrinsics:
    def __init__(self,
                 fx: float,
                 fy: float,
                 cx: float,
                 cy: float,
                 width: int = None,
                 height: int = None,
                 fov: float = None):
        self.fy = fy
        self.fx = fx
        self.cx = cx
        self.cy = cy
        self.fov = fov
        self.width = width
        self.height = height


class SequenceDataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self,
                 seq_dir: str,
                 intrinsics: CameraIntrinsics,
                 img_folder: str = "image_left",
                 pose_file: str = "pose_left.txt",
                 flow_folder: str = "flow",
                 transform=None):

        self._seq_dir = Path(f"{seq_dir}")
        self._intrinsics = intrinsics
        self._img_folder = self._seq_dir / img_folder
        self._pose_file = self._seq_dir / pose_file if (pose_file is not None and
                                                        pose_file.strip() != "") else None
        self._flow_folder = self._seq_dir / flow_folder if (flow_folder is not None and
                                                            flow_folder.strip() != "") else None
        self._transform = transform

        # RGB (must)
        files = os.listdir(f"{self._img_folder}")
        self._img_files = [self._img_folder / f
                           for f in files if (f.endswith('.png') or f.endswith('.jpg'))]
        self._img_files.sort()

        print(f'Find {len(self._img_files)} image files in {self._seq_dir}')

        # GT Pose (left) (optional, for training)
        self._motions = None
        if self._pose_file is not None:
            poselist = np.loadtxt(f"{self._pose_file}").astype(np.float32)
            assert (poselist.shape[1] == 7)  # position + quaternion

            poses = pos_quats2SEs(poselist)  # To SE3
            motions = pose2motion(poses)  # To relative motion (flatten SE3)
            self._motions = SEs2ses(motions).astype(np.float32)  # To se3
            assert (len(self._motions) == len(self._img_files)) - 1

        # GT Optical Flow (optional, for training)
        # Not using mask to make Optical flow NN learn more from masked (dynamic) area
        self._flow_files = None
        if self._flow_folder is not None:
            files = os.listdir(f"{self._flow_folder}")
            rx_flow = re.compile(r"flow\.npy$", re.MULTILINE | re.IGNORECASE)
            self._flow_files = [self._flow_folder / f
                                for f in files if rx_flow.search(f)]
            self._flow_files.sort()
            assert (len(self._flow_files) == len(self._img_files) - 1)

    def __len__(self):
        return len(self._img_files) - 1

    def __getitem__(self, idx):
        # RGB
        img1 = cv2.imread(f"{self._img_files[idx]}")
        img2 = cv2.imread(f"{self._img_files[idx + 1]}")

        # Intrinsics
        if (self._intrinsics.height is None or
            self._intrinsics.width is None or
            self._intrinsics.height <= 0 or
                self._intrinsics.width <= 0):
            h, w, _ = img1.shape
            self._intrinsics.height = h
            self._intrinsics.width = w

        intrinsicLayer = make_intrinsics_layer(
            self._intrinsics.width,
            self._intrinsics.height,
            self._intrinsics.fx,
            self._intrinsics.fy,
            self._intrinsics.cx,
            self._intrinsics.cy
        )

        res = {
            'img1': img1,
            'img2': img2,
            'intrinsic': intrinsicLayer,
        }

        # Optical flow
        if self._flow_files is not None:
            flowFile = self._flow_files[idx]
            res["flow"] = np.load(flowFile)

        # Transform before non-image type added to result
        if self._transform is not None:
            res = self._transform(res)

        if self._motions is not None:
            res['motion'] = self._motions[idx]

        return res

    def __str__(self) -> str:
        return f"{self._seq_dir}"


class TartanAirDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 intrinsics: CameraIntrinsics,
                 img_folder: str = "image_left",
                 pose_file: str = "pose_left.txt",
                 flow_folder: str = "flow",
                 transform=None,
                 difficulty: str = None,
                 environment: str = None):
        self._dataset_dir = Path(dataset_dir)
        self._intrinsics = intrinsics
        self._img_folder = img_folder
        self._pose_file = pose_file
        self._flow_folder = flow_folder
        self._transform = transform

        self._rx_seq = re.compile(r"^P\d{3}", re.MULTILINE)

        rows = []
        for root, dirs, files in os.walk(dataset_dir):
            for dir in dirs:
                # If it's sequenc dir
                if self._rx_seq.search(dir):
                    path = Path(os.path.join(root, dir))
                    rows.append({
                        "seq": SequenceDataset(path,
                                               intrinsics=intrinsics,
                                               img_folder=img_folder,
                                               pose_file=pose_file,
                                               flow_folder=flow_folder,
                                               transform=transform),
                        "diff": f"{path.parent.name}",
                        "env": f"{path.parent.parent.name}"
                    })
        self._all_seq_df = pd.DataFrame(rows)
        self._sub_seq_df = self._all_seq_df

        # Filter sequence
        self.difficulty = difficulty
        self.environment = environment

    @property
    def all(self) -> List[SequenceDataset]:
        return self._all_seq_df["seq"].tolist()

    @property
    def get_list(self) -> List[SequenceDataset]:
        return self._sub_seq_df["seq"].tolist()

    @property
    def difficulty(self):
        return self._difficulty

    @property
    def environment(self):
        return self._environment

    @difficulty.setter
    def difficulty(self, value: str):
        self._difficulty = value
        if self._difficulty is not None:
            self._sub_seq_df = self._all_seq_df.loc[self._all_seq_df["diff"]
                                                    == self._difficulty]

    @environment.setter
    def environment(self, value: str):
        self._environment = value
        if self._environment is not None:
            self._sub_seq_df = self._all_seq_df.loc[self._all_seq_df["env"]
                                                    == self._environment]

    def __len__(self) -> int:
        return len(self._sub_seq_df)

    def __getitem__(self, idx: int) -> SequenceDataset:
        return self._sub_seq_df.iloc[idx]["seq"]
