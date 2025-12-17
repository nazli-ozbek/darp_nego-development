import torch
import torch.nn as nn
import torch.optim as optim


class LogisticSwapModel(nn.Module):
    """
    Simple logistic regression model used to score client-agent swap proposals.
    The model consumes a flat feature vector and outputs the probability of
    a swap being accepted by all parties.
    """

    def __init__(self, input_dim: int, lr: float = 1e-2):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.linear(x))

    def train_batch(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        self.train()
        preds = self.forward(x_batch)
        loss = self.loss_fn(preds, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
