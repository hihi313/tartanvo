import argparse
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

from Datasets.tartanair import *
from Datasets.utils import (Compose, CropCenter, DownscaleFlow,
                            SampleNormalize, ToTensor, dataset_intrinsics,
                            load_kiiti_intrinsics)
from TartanVO import TartanVO

ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
DATASET_DIR = Path("/datasets/")


def get_args():
    parser = argparse.ArgumentParser(description="HRL")

    parser.add_argument(
        "--batch-size", type=int, default=1, help="batch size (default: 1)"
    )
    parser.add_argument(
        "--worker-num",
        type=int,
        default=None,
        help="data loader worker number (default: 1)",
    )
    parser.add_argument(
        "--image-width", type=int, default=640, help="image width (default: 640)"
    )
    parser.add_argument(
        "--image-height", type=int, default=448, help="image height (default: 448)"
    )
    parser.add_argument(
        "--model-name", default="", help='name of pretrained model (default: "")'
    )
    parser.add_argument(
        "--euroc",
        action="store_true",
        default=False,
        help="euroc test (default: False)",
    )
    parser.add_argument(
        "--kitti",
        action="store_true",
        default=False,
        help="kitti test (default: False)",
    )
    parser.add_argument(
        "--kitti-intrinsics-file",
        default="",
        help="kitti intrinsics file calib.txt (default: )",
    )
    parser.add_argument(
        "--test-dir",
        default="",
        help='test trajectory folder where the RGB images are (default: "")',
    )
    parser.add_argument(
        "--flow-dir",
        default="",
        help='test trajectory folder where the optical flow are (default: "")',
    )
    parser.add_argument(
        "--pose-file",
        default="",
        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")',
    )
    parser.add_argument(
        "--save-flow",
        action="store_true",
        default=False,
        help="save optical flow (default: False)",
    )

    args = parser.parse_args()

    return args


class PoseNormLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        # epsilon will auto move to the same device when compute
        self.epsilon = torch.tensor(
            epsilon, dtype=torch.float, requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, outputs, targets):
        # (From SE2se()): motion[0:3] = translation, motion[3:6] = rotation
        T_o = outputs[:, 0:3]
        R_o = outputs[:, 3:6]
        T_t = targets[:, 0:3]
        R_t = targets[:, 3:6]
        scale_o = torch.maximum(torch.norm(outputs, p=2, dim=1, keepdim=True),
                                self.epsilon)
        scale_t = torch.maximum(torch.norm(targets, p=2, dim=1, keepdim=True),
                                self.epsilon)
        T_loss = torch.norm((T_o / scale_o) - (T_t / scale_t),
                            p=2, dim=1, keepdim=True)
        R_loss = torch.norm(R_o - R_t, p=2, dim=1, keepdim=True)
        return T_loss + R_loss


def freeze_params(params: torch.Tensor):
    if isinstance(params, nn.Module):
        params = params.parameters()
    for p in params:
        p.requires_grad = False


if __name__ == "__main__":
    args = get_args()

    # load trajectory data from a folder
    datastr = "tartanair"
    if args.kitti:
        datastr = "kitti"
    elif args.euroc:
        datastr = "euroc"
    else:
        datastr = "tartanair"
    fx, fy, cx, cy = dataset_intrinsics(datastr)
    if args.kitti_intrinsics_file.endswith(".txt") and datastr == "kitti":
        fx, fy, cx, cy = load_kiiti_intrinsics(
            args.kitti_intrinsics_file
        )

    transform = Compose(
        [
            CropCenter((args.image_height, args.image_width)),
            DownscaleFlow(),
            SampleNormalize(flow_norm=20),  # pose normalize args is omitted
            ToTensor(),
        ]
    )

    intrinsics = CameraIntrinsics(fx, fy, cx, cy)
    tartanair_set = TartanAirDataset(f"{DATASET_DIR / 'TartanAir'}",
                                     intrinsics, transform=transform)
    all_seq_set = ConcatDataset(tartanair_set.all)
    train_set, valid_set = torch.utils.data.random_split(all_seq_set,
                                                         [0.8, 0.2])

    num_workers = os.cpu_count() if args.worker_num is None else args.worker_num
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers
    )

    epoch = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tartanvo = TartanVO(args.model_name)
    model = tartanvo.vonet
    # freeze_params(model.flowNet) # fix flow

    flow_criterion = nn.MSELoss(reduction="mean")
    pose_criterion = PoseNormLoss(1e-6)
    weight_lambda = 0.1
    optimizer = optim.SGD(model.parameters(), lr=1e-6)  # train pose

    writer = SummaryWriter(
        f"{LOG_DIR}/{datetime.now():%Y%m%d_%H%M}_{args.model_name}")

    model.train()
    best_loss = math.inf
    for ep in trange(epoch, desc="#Epoch", leave=False):
        flow_loss = pose_loss = loss = 0
        for sample in tqdm(train_loader, desc="#Batch", leave=False):
            # Data
            img0 = sample["img1"].to(device)
            img1 = sample["img2"].to(device)
            intrinsic = sample["intrinsic"].to(device)
            # Ground truth
            flow_gt = sample["flow"].to(device)  # N x C x H x W
            pose_gt = sample["motion"].to(device)  # N x 6

            # Forward
            flow, pose = model([img0, img1, intrinsic])

            # Loss
            # Pre-divide by batch size to make it numerically stable
            batch_flow_loss = flow_criterion(
                flow, flow_gt)  # /batche_size, scalar
            batch_pose_loss = pose_criterion(
                pose, pose_gt
            ).mean()  # /batche_size, scalar
            batch_loss = batch_pose_loss  # weight_lambda * batch_flow_loss  +

            # Total loss
            # Not accurate (a little bit) if all batch size are not the same
            flow_loss += batch_flow_loss.item()
            pose_loss += batch_pose_loss.item()
            loss += batch_loss.item()

            # Backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        # --- After trained an epoch ---
        # Log
        writer.add_scalar("Flow loss", flow_loss, ep)
        writer.add_scalar("Pose loss", pose_loss, ep)
        writer.add_scalar("Loss", loss, ep)
        print(
            f"Epoch {ep}: flow_loss={flow_loss}\tpose_loss={pose_loss}\tloss={loss}")

        # Validation
        # Select based on RPE if no RNN, else ATE
        model.eval()
        with torch.no_grad():
            # ses2poses_quat(np.array(motionlist)) # N*6->N*7 (0:3=t, 3:7=q)
            
            # Model checkpoint
            # if loss < best_loss:
            #     torch.save({
            #         "epoch": epoch,
            #         "model_state_dict": model.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #         "loss": loss,
            #     }, MODEL_DIR / f"tartanvo_{datetime.now():%Y%m%d_%H%M}.pkl")

    writer.flush()
    writer.close()
    print("\nFinished!")
