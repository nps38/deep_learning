"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

print("Time to train")

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from models import load_model, save_model
from datasets.road_dataset import RoadDataset

def train(
    model_name: str,
    n_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Load dataset
    train_dataset = RoadDataset(split="train")
    val_dataset = RoadDataset(split="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = load_model(model_name, n_track=10, n_waypoints=3).to(device)
    criterion = nn.MSELoss()  # Suitable for real-valued regression
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            track_left = batch["track_left"].to(device)  # (B, n_track, 2)
            track_right = batch["track_right"].to(device)  # (B, n_track, 2)
            waypoints = batch["waypoints"].to(device)  # (B, n_waypoints, 2)
            waypoints_mask = batch["waypoints_mask"].to(device)  # (B, n_waypoints)

            # Mask waypoints for clean targets
            waypoints = waypoints * waypoints_mask.unsqueeze(-1)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(track_left=track_left, track_right=track_right)
            loss = criterion(predictions, waypoints)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Average loss over the epoch
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

        # Logging
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the model
    model_path = save_model(model)
    print(f"Model saved at {model_path}")


if __name__ == "__main__":
    train(model_name="mlp_planner", n_epochs=20, batch_size=32, learning_rate=1e-3)

