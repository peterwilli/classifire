import argparse
import math
import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import torchmetrics
import os
from tqdm import tqdm
from imgaug import augmenters as iaa
import numpy as np
import random
from functools import partial
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import ImageOps
from model import Classifire
from pytorch_lightning.callbacks import LearningRateMonitor

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--enable_logging", type=bool, default=True)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout_p", type=float, default=0.01)
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args(args)


def get_datamodule(batch_size: int):
    train_transforms = transforms.Compose(
        [
            iaa.Resize({"shorter-side": (64, 128), "longer-side": "keep-aspect-ratio"}).augment_image,
            iaa.CropToFixedSize(width=64, height=64).augment_image,
            iaa.Sometimes(0.6, iaa.Sequential([
                iaa.flip.Fliplr(p=0.5),
                iaa.flip.Flipud(p=0.5),
                iaa.Sometimes(
                    0.5,
                    iaa.Sequential([
                        iaa.ShearX((-20, 20)),
                        iaa.ShearY((-20, 20))
                    ])
                ),
                iaa.Sometimes(0.2, iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ])),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.Sometimes(0.2, iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))),
                iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
                iaa.GaussianBlur(sigma=(0.0, 0.05)),
                iaa.MultiplyBrightness(mul=(0.65, 1.35)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            ], random_order=True)).augment_image,
            np.copy,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    test_transforms = transforms.Compose(
        [
            iaa.Resize({"shorter-side": (64, 128), "longer-side": "keep-aspect-ratio"}).augment_image,
            iaa.CropToFixedSize(width=64, height=64).augment_image,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    class ImageWeightDataset(Dataset):
        def __init__(self, path, transform):
            self.path = path
            self.files = os.listdir(self.path)
            self.transform = transform
            self.num_images = 2

        def _images_from_paths(self, paths):
            images = None
            for path in paths:
                image = Image.open(path).convert("RGB")
                image = ImageOps.exif_transpose(image)
                image = self.transform(np.array(image)).unsqueeze(0)
                if images is None:
                    images = image
                else:
                    images = torch.cat((images, image), 0)
            return images

        def _make_grid_of_paths(self, paths):
            images = self._images_from_paths(paths)
            grid = torchvision.utils.make_grid(images, nrow = 2, padding = 0)
            return grid

        def _get_image_paths_from_index(self, index):
            full_path = os.path.join(self.path, self.files[index])
            images_path = os.path.join(full_path, "concept_images")
            image_names = os.listdir(images_path)
            random.shuffle(image_names)
            against_itself = random.randint(0, 1) > 0.8
            image_names = image_names[:self.num_images]
            if against_itself:
                image_names = [image_names[0]] * self.num_images
            image_names_len = len(image_names)
            if image_names_len < self.num_images:
                for i in range(self.num_images - image_names_len):
                    image_names.append(image_names[i % image_names_len])
            return [os.path.join(images_path, image_name) for image_name in image_names]

        def __getitem__(self, index):
            generate_correct_classification = random.randint(0, 1) == 1
            if generate_correct_classification:
                paths = self._get_image_paths_from_index(index)
                images = self._images_from_paths(paths)
                # print("correct paths:", paths)
                return images, torch.tensor(1)
            else:
                paths_correct = self._get_image_paths_from_index(index)
                rnd_index = index
                while rnd_index == index:
                    rnd_index = random.randint(0, len(self.files) - 1)
                paths_incorrect = self._get_image_paths_from_index(rnd_index)
                unmatched_paths = [random.choice(paths_correct), random.choice(paths_incorrect)]
                images = self._images_from_paths(unmatched_paths)
                # print("incorrect paths:", unmatched_paths)
                return images, torch.tensor(0)
        
        def __len__(self):
            return len(self.files)

    class ImageWeights(pl.LightningDataModule):
        def __init__(self, data_folder: str, batch_size: int):
            super().__init__()
            self.num_workers = 16
            self.data_folder = data_folder
            self.batch_size = batch_size
            self.overfit = False
            self.num_samples = len(os.listdir(os.path.join(self.data_folder, "train")))
            if self.overfit:
                self.num_samples = 250
            
        def prepare_data(self):
            pass

        def setup(self, stage):
            pass
            
        def train_dataloader(self):
            dataset = ImageWeightDataset(os.path.join(self.data_folder, "train"), transform = train_transforms)
            if self.overfit:
                file_list = dataset.files[:1]
                print("Overfit! Using only:", file_list)
                dataset.files = file_list * 250
            return DataLoader(dataset, num_workers = self.num_workers, batch_size = self.batch_size, shuffle=True)

        def val_dataloader(self):
            return DataLoader(ImageWeightDataset(os.path.join(self.data_folder, "val"), transform = test_transforms), num_workers = self.num_workers, batch_size = self.batch_size)

        def test_dataloader(self):
            return DataLoader(ImageWeightDataset(os.path.join(self.data_folder, "test"), transform = test_transforms), num_workers = self.num_workers, batch_size = self.batch_size)

        def teardown(self, stage):
            pass
    
    dm = ImageWeights("./data", batch_size = batch_size)
    
    return dm

def test_train_images(dm):
    train_loader = dm.train_dataloader()
    count = 0
    transform = transforms.ToPILImage()
    for batch, _ in train_loader:
        for i in range(batch.shape[0]):
            image = batch[i, 0, ...]
            pil_image = transform(image)
            pil_image.save(f"test_{count}.png")
            count += 1
            if count > 100:
                return

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    pl.seed_everything(2)
    args = parse_args()
    
    # Add some dm attributes to args Namespace
    args.image_size = 128
    args.patch_size = 32
    args.input_shape = (3, 64, 64)

    # compute total number of steps
    batch_size = args.batch_size * args.gpus if args.gpus > 0 else args.batch_size
    dm = get_datamodule(batch_size = batch_size)
    # test_train_images(dm)
    args.steps = dm.num_samples // batch_size * args.max_epochs
    
    # Init Lightning Module
    lm = Classifire.load_from_checkpoint("lightning_logs/version_1/checkpoints/epoch=4999-step=75000.ckpt")
    # lm = Classifire(**vars(args))
    lm.train()
    # early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    args.callbacks = [lr_monitor]
    args.log_every_n_steps = 10

    # Init callbacks
    if args.enable_logging:
        pass
    else:
        args.checkpoint_callback = False
        args.logger = False
    
    # Set up Trainer
    trainer = pl.Trainer.from_argparse_args(args)
    
    # Train!
    trainer.fit(lm, dm)
