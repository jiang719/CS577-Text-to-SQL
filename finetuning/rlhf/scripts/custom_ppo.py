import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM

class PPO:
    def __init__(self, model: AutoModelForCausalLM, learning_rate: float):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.eps_clip = 0.2
        self.K_epochs = 4

    def update(self, old_probs, states, actions, rewards):
        for _ in range(self.K_epochs):
            # Get new probabilities and values
            logits, value = self.model(states)
            probs = F.softmax(logits, dim=-1)
            new_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            value = value.squeeze(1)

            # Calculate ratios
            ratios = new_probs / old_probs

            # Calculate advantages
            advantages = rewards - value.detach()

            # Calculate surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean()

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
