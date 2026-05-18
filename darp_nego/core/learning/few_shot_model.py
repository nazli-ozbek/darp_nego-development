import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np

class FewShotMediatorModel(nn.Module):
    """
    A small neural model that predicts the acceptability of a proposed outcome
    based on few-shot examples (previous negotiation rounds).
    """
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss() #binary cross-entropy loss
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return self.net(x)

    def train_on_batch(self, x_batch, y_batch):
        self.train()
        preds = self.forward(x_batch)
        loss = self.loss_fn(preds, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x).cpu().numpy()

    def clone_model(self):
        """Return a copy of the model (for inner-loop adaptation)."""
        input_dim = list(self.net.children())[0].in_features
        cloned = FewShotMediatorModel(input_dim=input_dim)
        cloned.load_state_dict(copy.deepcopy(self.state_dict()))
        return cloned

    def maml_inner_update(self, x_support, y_support, lr_inner=0.01):
        """
        Perform one inner-loop gradient step on the support set.
        Returns adapted parameters (state_dict).
        """
        preds = self.forward(x_support)
        loss = self.loss_fn(preds, y_support)
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        updated_params = {
            name: param - lr_inner * grad
            for ((name, param), grad) in zip(self.named_parameters(), grads)
        }
        return updated_params

    def maml_forward(self, x_query, params):
        """Forward pass with adapted parameters."""
        x = x_query
        for name, layer in self.net.named_children():
            if isinstance(layer, nn.Linear):
                weight = params[f'net.{name}.weight']
                bias = params[f'net.{name}.bias']
                x = nn.functional.linear(x, weight, bias)
            elif isinstance(layer, nn.ReLU):
                x = torch.relu(x)
            elif isinstance(layer, nn.Sigmoid):
                x = torch.sigmoid(x)
        return x

    def meta_train_step(self, tasks, lr_inner=0.01):
        """
        Perform one meta-training step over multiple tasks (negotiation cases).
        Each task = (x_support, y_support, x_query, y_query)
        """
        meta_loss = 0.0
        for (x_s, y_s, x_q, y_q) in tasks:
            updated_params = self.maml_inner_update(x_s, y_s, lr_inner=lr_inner)
            preds_q = self.maml_forward(x_q, updated_params)
            loss_q = self.loss_fn(preds_q, y_q)
            meta_loss += loss_q

        meta_loss = meta_loss / len(tasks)
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
        return meta_loss.item()
