import logging
import sys
import time
from pathlib import Path

import cpuinfo
import hydra
import omegaconf
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
import torch.optim.lr_scheduler as lr_scheduler

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from sim import utils
from sim import models
import torchvision.models as vision_models

logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval, use_cache=True)
OmegaConf.register_new_resolver(
    "generate_random_seed", utils.seeding.generate_random_seed, use_cache=True
)


@hydra.main(version_base=None, config_path="configs", config_name="train_explicit")
def main(config: DictConfig) -> None:
    utils.config.initialize_config(config)

    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Running with config: \n{OmegaConf.to_yaml(config, resolve=True)}")

    # Record total runtime.
    total_runtime = time.time()

    utils.seeding.seed_everything(config)

    if config.model == "ffnn":
        model = models.explicit.ffnn.FashionMNIST_FFNN(28 * 28, 10)
    elif config.model == "ResNet18":
        # model = vision_models.resnet18(pretrained=True)
        # model.avgpool = nn.AdaptiveAvgPool2d(1)
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, 200)
        model = models.explicit.cnn.resnet18()
    elif config.model == "ResNet10":
        model = models.explicit.cnn.resnet10()
    elif config.model == "MobileNetv3":
        model = vision_models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        if config.dataset == "TinyImageNet":
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, 200)
        elif config.dataset == "CIFAR10":
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, 10)
    elif config.model == "MobileNetv2":
        model = vision_models.mobilenet_v2(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 200)
    elif config.model == "EfficientNetB0":
        model = vision_models.efficientnet_b0(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 200)
    elif config.model == "MLPMixer":
        model = models.explicit.mixer.MLPMixer(
            img_size=28,
            patch_size=7,
            hidden_dim=32,
            token_dim=16,
            channel_dim=64,
            num_layers=2,
            num_classes=10,
        )
        # model = models.explicit.mixer.MLPMixer(
        #     in_channels=3,
        #     img_size=32,
        #     hidden_size=64,
        #     patch_size = 4,
        #     hidden_c = 512,
        #     hidden_s = 64,
        #     num_layers = 4,
        #     num_classes=10,
        #     drop_p=0.
        # )
    elif config.model == "SimCLR":
        # Prepare model
        # pre_model = models.explicit.cnn.SimCLR(models.explicit.cnn.resnet10().to(config.device), projection_dim=128).to(config.device)
        pre_model = models.explicit.cnn.SimCLR(
            vision_models.resnet18(pretrained=False).to(config.device), 
            projection_dim=128
        ).to(config.device)
        pre_model.load_state_dict(torch.load(config.backbone_ckpt))
        logger.info(f"Loaded pretrained feature extractor from {config.backbone_ckpt}")
        model = models.explicit.cnn.LinModel(pre_model.enc, feature_dim=pre_model.feature_dim, n_classes=10)
    elif config.model == "SimpleCNN":
        model = models.explicit.cnn.SimpleCNN()
    elif config.model == "SigmoidCNN":
        model = models.explicit.cnn.SigmoidCNN()
    else:
        raise NotImplementedError()
    # model = models.explicit.vit.ViT(img_size=28, patch_size=7, emb_size=32, num_heads=1, num_layers=1, num_classes=10).to(config.device)
    
    model = model.to(config.device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    train_loader, test_loader = utils.load_data.load_dataloader(config)

    if config.train.label_smoothing != 0.0:
        loss_fn = nn.CrossEntropyLoss(reduction="sum", label_smoothing=config.train.label_smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss(reduction="sum")

    if config.model=="SimCLR":
        train_params = [param for param in model.lin.parameters()]
    else:
        train_params = [param for param in model.parameters()]
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in train_params)}")

    # optimizer = optim.Adadelta(model.parameters(), lr=config.train.lr)
    if config.train.optimizer == "adam":
        optimizer = optim.Adam(train_params, lr=config.train.lr, betas=(0.9, 0.99), weight_decay=config.train.weight_decay)
    elif config.train.optimizer == "sgd":
        optimizer = optim.SGD(train_params, lr=config.train.lr, momentum=config.train.momentum, weight_decay=config.train.weight_decay)

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=config.train.lr_decay_per_epoch)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.num_epochs, eta_min=config.train.min_lr)

    models.explicit.utils.test(config, model, test_loader, loss_fn, 0)
    for epoch in range(1, config.train.num_epochs + 1):
        models.explicit.utils.train(
            config, model, train_loader, loss_fn, optimizer, epoch
        )
        models.explicit.utils.test(config, model, test_loader, loss_fn, epoch)
        scheduler.step()

    if config.save_model:
        torch.save(model.state_dict(), f"{config.model}_{config.dataset}.pt")
        print(f"Model successfully saved")

    # Record total runtime.
    total_runtime = time.time() - total_runtime
    # wandb.log({"total_runtime": total_runtime})


if __name__ == "__main__":
    main()
