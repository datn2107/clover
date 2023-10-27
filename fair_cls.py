import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F


class fair_classifier(torch.nn.Module):
    def __init__(self, config):
        super(fair_classifier, self).__init__()
        self.embedding_dim = config.embedding_dim * 4
        self.fc2_in_dim = config.first_fc_hidden_dim
        self.fc2_out_dim = config.num_gender

        self.fc1 = nn.Linear(self.fc2_in_dim // 2, self.fc2_in_dim // 2)
        self.fc2 = nn.Linear(self.fc2_in_dim // 2, self.fc2_out_dim)
        self.final_part = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, user_emb, training=True):
        gender_idx = x[:, 0]
        age_idx = x[:, 1]
        occupation_idx = x[:, 2]
        area_idx = x[:, 3]

        x = self.final_part(user_emb)
        loss = self.loss(x, gender_idx)

        return F.softmax(x, dim=1), gender_idx, loss


def get_user_embedding_adv(args, model, dataloader, user_index):
    user_embedding = np.zeros((len(user_index), args.first_fc_hidden_dim // 2))
    if args.train_adv:
        adv_now = model.task_adv[0]
        print(adv_now)
    else:
        adv_now = args.adv
    # only for inner_fc 1
    for weight in model.global_part.parameters():
        weight.fast = None
    for c, batch in tqdm(enumerate(dataloader)):
        x_spt = batch[0].cuda()  # Batch_size(1) x 交互物品个数 x features
        y_spt = batch[1].cuda()
        x_qry = batch[2].cuda()
        y_qry = batch[3].cuda()
        user_id = batch[4][0]
        index = int(np.argwhere(user_index == user_id))
        for i in range(x_spt.shape[0]):
            gender = x_qry[i].detach().cpu().numpy()[0, 21]
            if args.re_w:
                if args.dual:
                    if gender == 0:
                        adv = adv_now[0]
                    elif gender == 1:
                        adv = adv_now[1]
                else:
                    adv = adv_now
            else:
                if args.dual:
                    if gender == 0:
                        adv = adv_now[0]
                    elif gender == 1:
                        adv = adv_now[1]
                else:
                    adv = adv_now
            # -------------- inner update --------------
            fast_parameters = model.local_part.parameters()
            for weight in model.local_part.parameters():
                weight.fast = None
            # local max update
            fast_parameters_max = model.local_max_part.parameters()
            for weight in model.local_max_part.parameters():
                weight.fast = None
            for k in range(args.num_grad_steps_inner):
                output = model(x_spt[i], y_spt[i], [x_spt[i]], [y_spt[i]])
                logits, adv_loss = output[0], output[1]
                # local max update
                if args.loss == 1:
                    if args.disable_inner_max:
                        loss_max = F.cross_entropy(
                            logits, torch.unsqueeze(y_spt[i]).long() - 1
                        )
                    else:
                        loss_max = (
                            F.cross_entropy(
                                logits, torch.unsqueeze(y_spt[i]).long() - 1
                            )
                            - adv
                            * adv_loss**args.adv_loss_power
                            / args.adv_loss_power
                        )
                else:
                    if args.disable_inner_max:
                        loss_max = F.mse_loss(logits, y_spt[i])
                    else:
                        loss_max = (
                            F.mse_loss(logits, y_spt[i])
                            - adv
                            * adv_loss**args.adv_loss_power
                            / args.adv_loss_power
                        )
                grad_max = torch.autograd.grad(
                    loss_max, fast_parameters_max, create_graph=True
                )
                fast_parameters_max = []
                for k, weight in enumerate(model.local_max_part.parameters()):
                    if weight.fast is None:
                        weight.fast = (
                            weight - args.lr_inner * grad_max[k]
                        )  # create weight.fast
                    else:
                        weight.fast = weight.fast - args.lr_inner * grad_max[k]
                    fast_parameters_max.append(weight.fast)
            gender_idx = x_spt[0][:, 21]
            age_idx = x_spt[0][:, 20]
            occupation_idx = x_spt[0][:, 22]
            area_idx = x_spt[0][:, 23]
            # number of interactions x dim
            user_emb = (
                F.relu(
                    model.fc_user(
                        model.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
                    )
                )
                .detach()
                .cpu()
                .numpy()
            )
            user_embedding[index] = user_emb[0, :]
    return user_embedding


def get_user_embedding(args, model, dataloader, user_index):
    user_embedding = np.zeros((len(user_index), args.first_fc_hidden_dim // 2))
    for c, batch in tqdm(enumerate(dataloader)):
        x_spt = batch[0].cuda()  # Batch_size(1) x 交互物品个数 x features
        y_spt = batch[1].cuda()
        x_qry = batch[2].cuda()
        y_qry = batch[3].cuda()
        user_id = batch[4][0]
        index = int(np.argwhere(user_index == user_id))
        for i in range(x_spt.shape[0]):
            # -------------- inner update --------------
            fast_parameters = model.final_part.parameters()
            for weight in model.final_part.parameters():
                weight.fast = None
            for k in range(args.num_grad_steps_inner):
                # if args.adv:
                #     logits, adv_loss = model(x_spt[i])
                # else:
                output = model(x_spt[i], y_spt[i], [x_spt[i]], [y_spt[i]])
                logits = output[0]
                if args.loss == 1:
                    loss = F.cross_entropy(logits, torch.unsqueeze(y_spt[i]).long() - 1)
                else:
                    loss = F.mse_loss(logits, y_spt[i])
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                fast_parameters = []
                for k, weight in enumerate(model.final_part.parameters()):
                    if weight.fast is None:
                        weight.fast = (
                            weight - args.lr_inner * grad[k]
                        )  # create weight.fast
                    else:
                        weight.fast = weight.fast - args.lr_inner * grad[k]
                    fast_parameters.append(weight.fast)
            gender_idx = x_spt[0][:, 21]
            age_idx = x_spt[0][:, 20]
            occupation_idx = x_spt[0][:, 22]
            area_idx = x_spt[0][:, 23]
            # number of interactions x dim
            user_emb = (
                F.relu(
                    model.fc_user(
                        model.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
                    )
                )
                .detach()
                .cpu()
                .numpy()
            )
            user_embedding[index] = user_emb[0, :]
    return user_embedding
