import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import glob
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from animatableGaussian.utils import Camera, ModelParam


def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "body_pose": torch.from_numpy(smpl_params["body_pose"].astype(np.float32)),
        "global_orient": torch.from_numpy(smpl_params["global_orient"].astype(np.float32)),
        "transl": torch.from_numpy(smpl_params["transl"].astype(np.float32)),
    }


def time_encoding(t, dtype, max_freq=4):
    time_enc = torch.empty(max_freq * 2 + 1, dtype=dtype)

    for i in range(max_freq):
        time_enc[2 * i] = np.sin(2 ** i * torch.pi * t)
        time_enc[2 * i + 1] = np.cos(2 ** i * torch.pi * t)
    time_enc[max_freq * 2] = t
    return time_enc


def focal2tanfov(focal, pixels):
    return pixels/(2*focal)


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def getProjectionMatrix(tanHalfFovY, tanHalfFovX, znear=0.01, zfar=100.0):
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getCamPara(dataroot, opt):
    camera = np.load(os.path.join(dataroot, "cameras.npz"))
    intr = camera["intrinsic"]
    intr[:2] /= opt.downscale
    c2w = np.linalg.inv(camera["extrinsic"])
    height = int(camera["height"] / opt.downscale)
    width = int(camera["width"] / opt.downscale)
    focal_length_x = intr[0, 0]
    focal_length_y = intr[1, 1]
    tanFovY = focal2tanfov(focal_length_y, height)
    tanFovX = focal2tanfov(focal_length_x, width)
    camera_params = Camera()
    projmatrix = getProjectionMatrix(tanFovY, tanFovX)
    viewmatrix = torch.Tensor(c2w)

    camera_params.image_height = height
    camera_params.image_width = width
    camera_params.tanfovx = tanFovX
    camera_params.tanfovy = tanFovY
    camera_params.bg = torch.ones([3, height, width])
    camera_params.scale_modifier = 1.0
    camera_params.viewmatrix = viewmatrix.T
    camera_params.projmatrix = (projmatrix@viewmatrix).T
    camera_params.campos = torch.inverse(camera_params.viewmatrix)[3, :3]
    return camera_params


def read_imgs(dataroot, opt, resolution):
    start = opt.start
    end = opt.end + 1
    skip = opt.get("skip", 1)
    img_lists = sorted(
        glob.glob(f"{dataroot}/images/*.png"))[start:end:skip]
    msk_lists = sorted(
        glob.glob(f"{dataroot}/masks/*.npy"))[start:end:skip]
    frame_count = len(img_lists)
    imgs = []
    masks = []  # 新增：保存mask

    for index in tqdm(range(frame_count)):
        image_path = img_lists[index]
        img = Image.open(image_path)
        img = PILtoTorch(img, resolution)
        msk_path = msk_lists[index]
        msk = torch.from_numpy(np.load(msk_path).astype(np.float32))[
            None, None, ...]
        msk = F.interpolate(msk, scale_factor=1/opt.downscale,
                            mode='bilinear')
        
        # 保留原始mask
        masks.append(msk[0])
        # 应用mask到图像
        imgs.append(img[:3, ...] * msk[0] + 1-msk[0])
    return imgs, masks  # 返回图像和mask


def load_pose(dataroot, opt, split):
    start = opt.start
    end = opt.end + 1
    skip = opt.get("skip", 1)
    if os.path.exists(os.path.join(dataroot, f"poses/anim_nerf_{split}.npz")):
        cached_path = os.path.join(dataroot, f"poses/anim_nerf_{split}.npz")
    elif os.path.exists(os.path.join(dataroot, f"poses/{split}.npz")):
        cached_path = os.path.join(dataroot, f"poses/{split}.npz")
    else:
        cached_path = None

    if cached_path and os.path.exists(cached_path):
        print(f"[{split}] Loading from", cached_path)
        smpl_params = load_smpl_param(cached_path)
    else:
        print(f"[{split}] No optimized smpl found.")
        smpl_params = load_smpl_param(os.path.join(dataroot, f"poses.npz"))
        for k, v in smpl_params.items():
            if k != "betas":
                smpl_params[k] = v[start:end:skip]
    return smpl_params


class PeopleSnapshotDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, max_freq, split, opt):
        self.split = split
        self.max_freq = max_freq
        self.camera_params = getCamPara(dataroot, opt)
        self.imgs, self.masks = read_imgs(  # 接收mask
            dataroot, opt, (self.camera_params.image_width, self.camera_params.image_height))
        self.smpl_params = load_pose(dataroot, opt, split)
        
        # 用于时序一致性
        self.previous_rendered = None
        self.build_time_encoding()
    
    def build_time_encoding(self):
        time_encodings = {}
        for i in range(len(self)):
            t = i / len(self)
            time_encodings[t] = time_encoding(t, self.imgs[i].dtype, self.max_freq)
        self.time_encodings = time_encodings

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Returns:
            data["camera_params"] (vars(Camera)) : Input dict for gaussian rasterizer.
            data["model_param"] (vars(ModelParam)) : Input dict for a deformer model.
            data["gt"] (torch.Tensor[3, h, w]) : Ground truth image.
            data["mask"] (torch.Tensor[1, h, w]) : Foreground mask.
            data["time"] (torch.Tensor[max_freq * 2 + 1,]) : Time normalized to 0-1.
            data["index"] (int) : Frame index for temporal consistency.
        """
        t = index / self.__len__()

        smpl_param = ModelParam()
        smpl_param.global_orient = self.smpl_params["global_orient"][None, index]
        smpl_param.body_pose = self.smpl_params["body_pose"][index].reshape([
                                                                            1, -1, 3])
        smpl_param.transl = self.smpl_params["transl"][None, index]

        data = {"camera_params": vars(self.camera_params),
                "model_param": vars(smpl_param),
                "gt": self.imgs[index],
                "mask": self.masks[index],  # 新增mask
                # "time":  self.time_encodings[t],
                "time":  time_encoding(t, self.imgs[index].dtype, self.max_freq),
                "index": index}  # 新增index用于时序一致性
        return data


def my_collate_fn(batch):
    return batch[0]


class PrefetchLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
    
    def move_to_device(self, batch, device, non_blocking=True):
        if torch.is_tensor(batch):
            return batch.to(device, non_blocking=non_blocking)
        elif isinstance(batch, dict):
            return {k: self.move_to_device(v, device, non_blocking) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self.move_to_device(v, device, non_blocking) for v in batch)
        else:
            return batch

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True
        for next_batch in self.loader:
            if first:
                with torch.cuda.stream(stream):
                    next_batch = self.move_to_device(next_batch, self.device, non_blocking=True)
                if not first:
                    torch.cuda.current_stream().wait_stream(stream)
                    yield batch
                else:
                    first = False
            else:
                yield batch
            batch = next_batch
        yield batch

    def __len__(self):
        return len(self.loader)


class PeopleSnapshotDataModule(pl.LightningDataModule):
    def __init__(self, num_workers, opt, train=True, **kwargs):
        super().__init__()
        if train:
            splits = ["train", "val"]
        else:
            splits = ["test"]
        for split in splits:
            print(f"loading {split}set...")
            dataset = PeopleSnapshotDataset(
                opt.dataroot, opt.max_freq, split, opt.get(split))
            setattr(self, f"{split}set", dataset)
        self.num_workers = num_workers

    def train_dataloader(self):
        if hasattr(self, "trainset"):
            # # for debug
            # return DataLoader(self.trainset,
            #                   shuffle=True,
            #                   pin_memory=True,
            #                   batch_size=1,
            #                   persistent_workers=False,
            #                   num_workers=0,
            #                   collate_fn=my_collate_fn)
            loader = DataLoader(self.trainset,
                              shuffle=True,
                              pin_memory=True,
                              batch_size=1,
                              persistent_workers=True,
                              num_workers=self.num_workers,
                              collate_fn=my_collate_fn)
            return loader
            return PrefetchLoader(loader, "cuda")
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valset"):
            # # for debug
            # return DataLoader(self.valset,
            #                   shuffle=False,
            #                   pin_memory=True,
            #                   batch_size=1,
            #                   persistent_workers=False,
            #                   num_workers=0,
            #                   collate_fn=my_collate_fn)
            return DataLoader(self.valset,
                              shuffle=False,
                              pin_memory=True,
                              batch_size=1,
                              persistent_workers=True,
                              num_workers=self.num_workers,
                              collate_fn=my_collate_fn)
        else:
            return super().val_dataloader()

    def test_dataloader(self):
        if hasattr(self, "testset"):
            return DataLoader(self.testset,
                              shuffle=False,
                              pin_memory=True,
                              batch_size=1,
                              persistent_workers=True,
                              num_workers=self.num_workers,
                              collate_fn=my_collate_fn)
        else:
            return super().test_dataloader()
