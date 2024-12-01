"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

print("Time to train")

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from .models import load_model, save_model
from .datasets.road_dataset import load_data
import argparse

def train(
    model_name: str,
    transform_pipeline: str,
    num_workers: int,
    lr: float,
    batch_size: int,
    num_epoch: int,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    train_loader = load_data(
        dataset_path="drive_data/train",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = load_data(
        dataset_path="drive_data/val",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    # Initialize model
    model = load_model(model_name, n_track=10, n_waypoints=3, dropout_rate=0.1).to(device)
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    is_cnn_planner = model_name.lower() == "cnn_planner"

    # Training loop
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0
        total_longitudinal_error = 0.0
        total_lateral_error = 0.0

        for batch in train_loader:
            if is_cnn_planner:
                image = batch["image"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)
            else:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

            waypoints = waypoints * waypoints_mask.unsqueeze(-1)

            optimizer.zero_grad()
            if is_cnn_planner:
                predictions = model(image=image)
            else:
                predictions = model(track_left=track_left, track_right=track_right)

            # Compute loss (e.g., MSE)
            loss = criterion(predictions, waypoints)
            loss.backward()
            optimizer.step()

            # Calculate longitudinal and lateral errors for this batch
            longitudinal_error = torch.abs(predictions[:, :, 0] - waypoints[:, :, 0])
            lateral_error = torch.abs(predictions[:, :, 1] - waypoints[:, :, 1])

            total_longitudinal_error += longitudinal_error.sum().item()
            total_lateral_error += lateral_error.sum().item()
            train_loss += loss.item()

        # Average loss over the epoch
        train_loss /= len(train_loader)
        total_longitudinal_error /= len(train_loader.dataset)
        total_lateral_error /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_longitudinal_error = 0.0
        val_lateral_error = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if is_cnn_planner:
                    image = batch["image"].to(device)
                    waypoints = batch["waypoints"].to(device)
                    waypoints_mask = batch["waypoints_mask"].to(device)
                else:
                    track_left = batch["track_left"].to(device)
                    track_right = batch["track_right"].to(device)
                    waypoints = batch["waypoints"].to(device)
                    waypoints_mask = batch["waypoints_mask"].to(device)

                # waypoints = waypoints * waypoints_mask.unsqueeze(-1)
                if is_cnn_planner:
                    predictions = model(image=image)
                else:
                    predictions = model(track_left=track_left, track_right=track_right)
                    
                masked_predictions = predictions * waypoints_mask.unsqueeze(-1)
                masked_waypoints = waypoints * waypoints_mask.unsqueeze(-1)

                # loss = criterion(predictions, waypoints)
                loss = criterion(masked_predictions, masked_waypoints)

                # Calculate longitudinal and lateral errors for validation
                longitudinal_error = torch.abs(predictions[:, :, 0] - waypoints[:, :, 0])
                lateral_error = torch.abs(predictions[:, :, 1] - waypoints[:, :, 1])

                val_longitudinal_error += longitudinal_error.sum().item()
                val_lateral_error += lateral_error.sum().item()
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_longitudinal_error /= len(val_loader.dataset)
        val_lateral_error /= len(val_loader.dataset)

        # Logging
        print(f"Epoch {epoch+1}/{num_epoch}, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train Longitudinal Error: {total_longitudinal_error:.4f}, "
            f"Train Lateral Error: {total_lateral_error:.4f}, "
            f"Val Longitudinal Error: {val_longitudinal_error:.4f}, "
            f"Val Lateral Error: {val_lateral_error:.4f}")

        # Save the model
        model_path = save_model(model)
        # print(f"Model saved at {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transform_pipeline", type=str, default="state_only")

    args = parser.parse_args()
    train(
        model_name=args.model_name,
        transform_pipeline=args.transform_pipeline,
        num_workers=4,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epoch=args.num_epoch,
    )
