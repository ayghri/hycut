import argparse
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

from simclr import SimCLR, simclr_loss_func
from data import get_dataloaders
from utils import LARS, AverageMeter, compute_knn


def parse_args():
    parser = argparse.ArgumentParser(description="SimCLR Training")
    parser.add_argument(
        "--dataset", default="cifar10", type=str, help="cifar10 or cifar100"
    )
    parser.add_argument(
        "--backbone", default="resnet18", type=str, help="resnet18 or resnet50"
    )
    parser.add_argument(
        "--batch_size", default=256, type=int, help="Batch size"
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs"
    )
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument(
        "--weight_decay", default=1e-3, type=float, help="Weight decay"
    )
    parser.add_argument(
        "--temperature", default=0.1, type=float, help="Temperature"
    )
    parser.add_argument(
        "--proj_hidden_dim", default=2048, type=int, help="Projector hidden dim"
    )
    parser.add_argument(
        "--proj_output_dim", default=128, type=int, help="Projector output dim"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of workers"
    )
    parser.add_argument(
        "--log_dir", default="./logs", type=str, help="Log directory"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="./checkpoints",
        type=str,
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--eval_freq", default=5, type=int, help="Evaluation frequency"
    )
    parser.add_argument(
        "--resume", default="", type=str, help="Path to checkpoint to resume"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    writer = SummaryWriter(args.log_dir)

    # Data
    train_loader, test_loader = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers
    )

    # Model
    model = SimCLR(
        args.backbone, args.proj_hidden_dim, args.proj_output_dim, cifar=True
    )
    model = model.to(device)
    # model.load_state_dict(torch.load("./checkpoints/checkpoint_epoch_50.pth"))

    # Optimizer
    # SimCLR uses LARS with learning rate scaling: lr = base_lr * batch_size / 256
    lr = args.lr
    # * args.batch_size / 256
    # optimizer = LARS(
    #     model.parameters(),
    #     lr=lr,
    #     weight_decay=args.weight_decay,
    #     exclude_bias_n_norm=True
    # )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5, last_epoch=-1
    )

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")

        end = time.time()
        for i, (img1, img2, _) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        ):
            img1, img2 = img1.to(device), img2.to(device)

            # Forward
            out1 = model(img1)
            out2 = model(img2)

            z1 = out1["z"]
            z2 = out2["z"]

            # Concatenate views: [z1_1, z1_2, ..., z2_1, z2_2, ...]
            # My loss function expects [v1_all, v2_all] stacked vertically?
            # Let's check simclr_loss_func in simclr.py
            # It assumes z is (2*N, D)
            # And pos_idx = (pos_idx + N) % batch_size
            # This implies the first N are view1, and the second N are view2.

            z = torch.cat([z1, z2], dim=0)

            loss = simclr_loss_func(z, args.temperature)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), img1.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

        scheduler.step()

        # Logging
        writer.add_scalar("train_loss", losses.avg, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        print(f"Epoch {epoch + 1}: Loss {losses.avg:.4f}")

        # Save checkpoint
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(
            save_dict, os.path.join(args.checkpoint_dir, "checkpoint_last.pth")
        )

        if (epoch + 1) % args.eval_freq == 0:
            print("Evaluating...")
            # For kNN, we need a train loader that returns single images, not pairs
            # But our train_loader returns pairs.
            # We can just use the first view for feature extraction.
            # compute_knn handles this.

            acc1, acc5 = compute_knn(model, train_loader, test_loader, device)
            print(f"kNN Accuracy: Top1: {acc1:.2f}, Top5: {acc5:.2f}")
            writer.add_scalar("knn_acc1", acc1, epoch)
            writer.add_scalar("knn_acc5", acc5, epoch)

            torch.save(
                save_dict,
                os.path.join(
                    args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"
                ),
            )


if __name__ == "__main__":
    main()
