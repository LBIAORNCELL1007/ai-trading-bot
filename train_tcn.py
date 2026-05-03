import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression


# 1. Data Preparation
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length=60):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx : idx + self.sequence_length], dtype=torch.float32),
            torch.tensor(self.y[idx + self.sequence_length - 1], dtype=torch.float32),
        )


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute any of the four required features that don't already exist in df:
       close_fd_04, rsi_14, MACD_12_26_9, atr_14

    These names match what the rest of the pipeline produces (see
    apply_fracdiff.py for `close_fd_04`).  Without this, TRAINING SILENTLY
    FAILED — `df.dropna(subset=features+[target])` removed every row when
    columns were missing, leaving zero training samples.
    """
    out = df.copy()

    # close_fd_04 — fractional differentiation, d=0.4
    if "close_fd_04" not in out.columns:
        if "close_fd" in out.columns:
            out["close_fd_04"] = out["close_fd"]
        elif "close" in out.columns:
            # Lazy import to avoid hard dependency on frac_diff during inference
            try:
                from frac_diff import frac_diff_ffd

                out["close_fd_04"] = frac_diff_ffd(out[["close"]], d=0.4)["close"]
            except Exception as e:
                raise RuntimeError(
                    f"close_fd_04 missing and frac_diff_ffd unavailable: {e}"
                )

    # rsi_14
    if "rsi_14" not in out.columns:
        if "close" not in out.columns:
            raise RuntimeError("Cannot compute rsi_14: 'close' column not present.")
        delta = out["close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        out["rsi_14"] = 100 - (100 / (1 + rs))
        out["rsi_14"] = out["rsi_14"].fillna(50.0)

    # MACD_12_26_9 (histogram)
    if "MACD_12_26_9" not in out.columns:
        if "close" not in out.columns:
            raise RuntimeError(
                "Cannot compute MACD_12_26_9: 'close' column not present."
            )
        ema12 = out["close"].ewm(span=12, adjust=False).mean()
        ema26 = out["close"].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        out["MACD_12_26_9"] = (macd_line - signal_line).fillna(0.0)

    # atr_14 (Wilder smoothing)
    if "atr_14" not in out.columns:
        if not all(c in out.columns for c in ("high", "low", "close")):
            raise RuntimeError("Cannot compute atr_14: high/low/close required.")
        prev_close = out["close"].shift(1)
        tr = pd.concat(
            [
                (out["high"] - out["low"]).abs(),
                (out["high"] - prev_close).abs(),
                (out["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr_14"] = (
            tr.ewm(alpha=1 / 14, adjust=False).mean().fillna(method="bfill").fillna(0.0)
        )

    return out


def load_and_prepare_data(csv_path, sequence_length=60):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Auto-compute missing features so the pipeline doesn't silently train
    # on zero rows (the previous failure mode that produced models trained
    # entirely on the random-fallback dummy data).
    df = _ensure_features(df)

    # NOTE: was 'close_fd' (legacy name) — actual column produced by
    # apply_fracdiff.py is 'close_fd_04'.  Updated in `_ensure_features`
    # to alias either way.
    features = ["close_fd_04", "rsi_14", "MACD_12_26_9", "atr_14"]
    target = "tbm_label"

    # Ensure no NaNs exist to prevent training errors
    df = df.dropna(subset=features + [target])
    if len(df) == 0:
        raise RuntimeError(
            "Zero usable rows after feature computation.  Check that the "
            "input CSV contains 'close' (and ideally 'high','low','tbm_label')."
        )

    X = df[features].values
    y = df[target].values

    # Chronological Split: 70% Train, 15% Validation, 15% Test
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train_raw, y_train = X[:train_end], y[:train_end]
    X_val_raw, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test_raw, y_test = X[val_end:], y[val_end:]

    # Standardize features (Fit only on Train, transform Val and Test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Calculate Class Weights based on training set imbalance (roughly 62/38)
    num_positives = np.sum(y_train == 1)
    num_negatives = len(y_train) - num_positives
    pos_weight = num_negatives / (num_positives + 1e-8)

    train_dataset = TimeSeriesDataset(X_train, y_train, sequence_length)
    val_dataset = TimeSeriesDataset(X_val, y_val, sequence_length)
    test_dataset = TimeSeriesDataset(X_test, y_test, sequence_length)

    # No shuffling for time-series validation/test data (we keep train shuffle=False for purity as well,
    # though shuffling within batches can sometimes be used. Per requirements, we maintain order).
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader, pos_weight


# 2. TCN Architecture
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, **kwargs):
        super(CausalConv1d, self).__init__()
        # Causal padding = (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            **kwargs,
        )

    def forward(self, x):
        x = self.conv(x)
        # Remove trailing elements to maintain causal alignment
        if self.padding != 0:
            x = x[:, :, : -self.padding]
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ResidualBlock, self).__init__()
        self.causal_conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size=3, dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.causal_conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size=3, dilation=dilation
        )
        self.relu2 = nn.ReLU()

        # Match dimensions if in_channels != out_channels
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.causal_conv1(x)
        out = self.relu1(out)
        out = self.causal_conv2(out)
        out = self.relu2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu3(out + res)


class TCN(nn.Module):
    def __init__(self, input_size, num_channels, sequence_length):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # 4 Dilated layers (dilation = 1, 2, 4, 8) as requested
            dilation_size = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(ResidualBlock(in_channels, out_channels, dilation_size))

        self.network = nn.Sequential(*layers)

        # Output shape after convolutions: (batch_size, out_channels, sequence_length)
        # We flatten it for the FC layer
        self.fc = nn.Linear(num_channels[-1] * sequence_length, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape expected by DataLoader: (batch_size, sequence_length, features)
        # Conv1d expects: (batch_size, channels/features, sequence_length)
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out.reshape(out.size(0), -1)  # Flatten
        out = self.fc(out)
        return self.sigmoid(out).squeeze(-1)


# 4. Training Loop
def train_model(
    model, train_loader, val_loader, pos_weight, num_epochs=100, patience=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)

    # 3. Anti-Lazy Optimization: BCELoss with positive class weights
    # nn.BCELoss(reduction='none') allows us to multiply sample weights per element before averaging.
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            # Apply class weights: weight is `pos_weight` if target is 1, else 1.0
            sample_weights = torch.where(
                y_batch == 1.0,
                torch.tensor(pos_weight, device=device),
                torch.tensor(1.0, device=device),
            )

            loss_raw = criterion(outputs, y_batch)
            loss = (loss_raw * sample_weights).mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)

                sample_weights = torch.where(
                    y_batch == 1.0,
                    torch.tensor(pos_weight, device=device),
                    torch.tensor(1.0, device=device),
                )
                loss_raw = criterion(outputs, y_batch)
                loss = (loss_raw * sample_weights).mean()

                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch + 1:03d}/{num_epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "tcn_global_model.pth")
            print("  --> Validation loss improved. Saved tcn_global_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break


# 5. Evaluation & Platt Scaling
def evaluate_and_calibrate(model, val_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("tcn_global_model.pth"))
    model.to(device)
    model.eval()

    print("\n--- Calibration ---")
    print("Extracting validation probabilities to fit Platt Scaler...")
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch.to(device))
            val_probs.extend(outputs.cpu().numpy())
            val_targets.extend(y_batch.numpy())

    val_probs = np.array(val_probs).reshape(-1, 1)
    val_targets = np.array(val_targets)

    # Fit Platt Scaler (Logistic Regression) on validation dataset
    platt_scaler = LogisticRegression()
    platt_scaler.fit(val_probs, val_targets)
    print("Platt Scaler fitted successfully.")

    print("\n--- Test Set Evaluation ---")
    test_probs = []
    test_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.to(device))
            test_probs.extend(outputs.cpu().numpy())
            test_targets.extend(y_batch.numpy())

    test_probs = np.array(test_probs).reshape(-1, 1)
    test_targets = np.array(test_targets)

    # Calibrate probabilities using the Platt Scaler
    calibrated_probs = platt_scaler.predict_proba(test_probs)[:, 1]

    # Calculate metrics
    auc_score = roc_auc_score(test_targets, calibrated_probs)

    # Default 0.5 threshold on calibrated probabilities
    preds = (calibrated_probs >= 0.5).astype(int)
    precision = precision_score(test_targets, preds, zero_division=0)
    recall = recall_score(test_targets, preds, zero_division=0)

    print(f"ROC-AUC Score: {auc_score:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")

    return platt_scaler


if __name__ == "__main__":
    csv_file = "btc_training_dataset.csv"

    # SAFETY: refuse to train on dummy/random data. Production models must be
    # trained on real OHLCV+labels produced by build_dataset.py + tbm_labeler.py.
    # Silent dummy-data fallback was a critical bug: it produced a model file
    # indistinguishable from a real one but with zero predictive signal.
    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"Training dataset '{csv_file}' not found. Generate it first via "
            "`python build_dataset.py` followed by `python tbm_labeler.py`. "
            "Refusing to train on synthetic random data."
        )

    seq_len = 60
    train_loader, val_loader, test_loader, pos_weight = load_and_prepare_data(
        csv_file, sequence_length=seq_len
    )

    print(f"Calculated positive class weight (0 vs 1 imbalance): {pos_weight:.4f}")

    input_size = 4  # Number of features
    # 4 dilated layers (dilation 1, 2, 4, 8) -> 4 blocks. Output channels configuration:
    num_channels = [32, 32, 32, 32]

    model = TCN(
        input_size=input_size, num_channels=num_channels, sequence_length=seq_len
    )

    train_model(
        model, train_loader, val_loader, pos_weight, num_epochs=100, patience=10
    )

    evaluate_and_calibrate(model, val_loader, test_loader)
