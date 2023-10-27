import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import  re
import datetime
import pandas as pd
import json
from tqdm import tqdm


def item_converting(row, rate_list, genre_list, director_list, actor_list):
    rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()
    genre_idx = -1 * torch.ones(1, 7).long()
    for i,genre in enumerate(str(row['genre']).split(", ")):
        idx = genre_list.index(genre)
        genre_idx[0, i] = idx
    director_idx = -1*torch.ones(1, 12).long()
    for i,director in enumerate(str(row['director']).split(", ")):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, i] = idx
    actor_idx = -1*torch.ones(1, 4).long()
    for i,actor in enumerate(str(row['actors']).split(", ")):
        idx = actor_list.index(actor)
        actor_idx[0, i] = idx
    return torch.cat((rate_idx, genre_idx, director_idx, actor_idx), 1)


def user_converting(row, gender_list, age_list, occupation_list, zipcode_list):
    gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
    zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()
    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)


def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_


class Metamovie(Dataset):
    def __init__(self, args, partition='train', test_way=None, path=None):
        super(Metamovie, self).__init__()
        self.partition = partition
        self.adv = args.adv
        self.seed = args.seed
        dataset_path = "ml-100k"
        if partition == 'train':
            self.state = 'user_warm_state'
        else:
            if test_way is not None:
                if test_way == 'old':
                    self.state = 'user_warm_state'
                elif test_way == 'new_user_valid':
                    self.state = 'user_cold_state_valid'
                elif test_way == 'new_user_test':
                    self.state = 'user_cold_state_test'
                elif test_way == 'new_item':
                    self.state = 'item_cold_state'
                else:
                    self.state = 'user_and_item_cold_state'
        print(self.state)
        with open("{}/{}.json".format(dataset_path, self.state), encoding="utf-8") as f:
            # str inside
            self.dataset_split = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split_y = json.loads(f.read())
        length = len(self.dataset_split.keys())
        self.final_index = []
        for _, user_id in tqdm(enumerate(list(self.dataset_split.keys()))):
            u_id = int(user_id)
            seen_movie_len = len(self.dataset_split[str(u_id)])
            if seen_movie_len < 13 or seen_movie_len > 100:
                continue
            else:
                self.final_index.append(user_id)
        with open("{}/{}.json".format(dataset_path, "movie_profile"), encoding="utf-8") as f:
            # str inside
            self.movie_dict = json.loads(f.read())
        with open("{}/{}.json".format(dataset_path, "user_profile"), encoding="utf-8") as f:
            self.user_dict = json.loads(f.read())

    def __getitem__(self, item):
        user_id = self.final_index[item]
        u_id = int(user_id)
        seen_movie_len = len(self.dataset_split[str(u_id)])
        indices = list(range(seen_movie_len))
        # random.seed(53)
        if self.state=="warm_state":
            random.shuffle(indices)
        tmp_x = np.array(self.dataset_split[str(u_id)])
        tmp_y = np.array(self.dataset_split_y[str(u_id)])

        support_x_app = None
        for m_id in tmp_x[indices[:-10]]:
            m_id = str(m_id)
            u_id = str(user_id)
            tmp_x_converted = np.expand_dims(np.concatenate((self.movie_dict[str(m_id)], self.user_dict[str(u_id)]), 0),0)
            try:
                support_x_app = np.concatenate((support_x_app, tmp_x_converted), 0)
            except:
                support_x_app = tmp_x_converted
        query_x_app = None
        support_items = np.array([int(m_id) for m_id in tmp_x[indices[:-10]]])
        test_items = np.array([int(m_id) for m_id in tmp_x[indices[-10:]]])
        for m_id in tmp_x[indices[-10:]]:
            m_id = str(m_id)
            u_id = str(user_id)
            tmp_x_converted = np.expand_dims(np.concatenate((self.movie_dict[str(m_id)], self.user_dict[str(u_id)]), 0),0)
            try:
                query_x_app = np.concatenate((query_x_app, tmp_x_converted), 0)
            except:
                query_x_app = tmp_x_converted
        support_x_app = torch.tensor(support_x_app).long()
        query_x_app = torch.tensor(query_x_app).long()
        support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])
        query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])
        # user_id and tmp_x is str
        return support_x_app, support_y_app.view(-1,1), query_x_app, query_y_app.view(-1,1), user_id, test_items

    def __len__(self):
        return len(self.final_index)


class Metamovie_fair(Dataset):
    def __init__(self, args, partition='train', test_way=None, path=None):
        super(Metamovie_fair, self).__init__()
        self.partition = partition
        self.adv = args.adv
        self.seed = args.seed
        dataset_path = "ml-100k"
        if partition == 'train':
            self.state = 'user_warm_state'
        else:
            if test_way is not None:
                if test_way == 'old':
                    self.state = 'user_warm_state'
                elif test_way == 'new_user_valid':
                    self.state = 'user_cold_state_valid'
                elif test_way == 'new_user_test':
                    self.state = 'user_cold_state_test'
                elif test_way == 'new_item':
                    self.state = 'item_cold_state'
                else:
                    self.state = 'user_and_item_cold_state'
        print(self.state)
        with open("{}/{}.json".format(dataset_path, self.state), encoding="utf-8") as f:
            # str inside
            self.dataset_split = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split_y = json.loads(f.read())
        length = len(self.dataset_split.keys())
        self.final_index = []
        for _, user_id in tqdm(enumerate(list(self.dataset_split.keys()))):
            u_id = int(user_id)
            seen_movie_len = len(self.dataset_split[str(u_id)])
            if seen_movie_len < 13 or seen_movie_len > 100:
                continue
            else:
                self.final_index.append(user_id)
        with open("{}/{}.json".format(dataset_path, "movie_profile"), encoding="utf-8") as f:
            # str inside
            self.movie_dict = json.loads(f.read())
        with open("{}/{}.json".format(dataset_path, "user_profile"), encoding="utf-8") as f:
            self.user_dict = json.loads(f.read())
        self.user_embedding = None


    def __getitem__(self, item):
        user_id = self.final_index[item]
        user_embedding = self.user_embedding[item]
        u_id = int(user_id)
        user_profile = self.user_dict[str(u_id)]
        return user_profile.view(-1), torch.tensor(user_embedding).to(torch.float32)

    def __len__(self):
        return len(self.final_index)
