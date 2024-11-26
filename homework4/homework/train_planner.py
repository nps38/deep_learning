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
from .datasets.road_dataset import RoadDataset
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
    train_dataset = RoadDataset(split="train", transform=transform_pipeline)
    val_dataset = RoadDataset(split="val", transform=transform_pipeline)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize model
    model = load_model(model_name, n_track=10, n_waypoints=3).to(device)
    criterion = nn.MSELoss()  # Suitable for real-valued regression
    optimizer = Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epoch):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            waypoints = waypoints * waypoints_mask.unsqueeze(-1)

            optimizer.zero_grad()
            predictions = model(track_left=track_left, track_right=track_right)
            loss = criterion(predictions, waypoints)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                waypoints = waypoints * waypoints_mask.unsqueeze(-1)
                predictions = model(track_left=track_left, track_right=track_right)
                loss = criterion(predictions, waypoints)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{num_epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    model_path = save_model(model, name=f"{model_name}_final")
    print(f"Model saved at {model_path}")


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
