import torch
import torch.nn as nn
from torch.nn import functional as F


class fair_classifier(torch.nn.Module):
    def __init__(self, config):
        super(fair_classifier, self).__init__()
        self.embedding_dim = config.embedding_dim * 4
        self.fc2_in_dim = config.first_fc_hidden_dim
        self.fc2_out_dim = config.num_gender

        self.fc1 = nn.Linear(self.fc2_in_dim//2, self.fc2_in_dim//2)
        self.fc2 = nn.Linear(self.fc2_in_dim//2, self.fc2_out_dim)
        self.final_part = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, user_emb, training = True):
        gender_idx = x[:, 0]
        age_idx = x[:, 1]
        occupation_idx = x[:, 2]
        area_idx = x[:, 3]

        x = self.final_part(user_emb)
        loss = self.loss(x, gender_idx)

        return F.softmax(x, dim=1), gender_idx, loss
