import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch

# Атрибуты действий пользователей в соцсетях
attributes = ['likes', 'comments', 'shares', 'time_spent', 'page_views', 'clicks', 'scroll_depth']

class UserBehaviorDataset(Dataset):
    def __init__(self, eval_length=36, target_dim=36, mode="train", validindex=0):
        self.eval_length = eval_length
        self.target_dim = target_dim

        path = "./data/user_behavior/behavior_meanstd.pk"
        with open(path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)

        if mode == "train":
            user_groups = [1, 2, 4, 5, 7, 8, 10, 11]
            flag_for_histmask = [0, 1, 0, 1, 0, 1, 0, 1]
            user_groups.pop(validindex)
            flag_for_histmask.pop(validindex)
        elif mode == "valid":
            user_groups = [1, 2, 4, 5, 7, 8, 10, 11]
            user_groups = user_groups[validindex : validindex + 1]
        elif mode == "test":
            user_groups = [3, 6, 9, 12]
        self.user_groups = user_groups

        self.observed_data = []
        self.observed_mask = []
        self.gt_mask = []
        self.index_group = []
        self.position_in_group = []
        self.valid_for_histmask = []
        self.use_index = []
        self.cut_length = []

        df = pd.read_csv("./data/user_behavior/behavior_data.csv", index_col="datetime", parse_dates=True)
        df_gt = pd.read_csv("./data/user_behavior/behavior_gt.csv", index_col="datetime", parse_dates=True)
        for i in range(len(user_groups)):
            current_df = df[df.index.month == user_groups[i]]
            current_df_gt = df_gt[df_gt.index.month == user_groups[i]]
            current_length = len(current_df) - eval_length + 1

            last_index = len(self.index_group)
            self.index_group += np.array([i] * current_length).tolist()
            self.position_in_group += np.arange(current_length).tolist()
            if mode == "train":
                self.valid_for_histmask += np.array([flag_for_histmask[i]] * current_length).tolist()

            c_mask = 1 - current_df.isnull().values
            c_gt_mask = 1 - current_df_gt.isnull().values
            c_data = ((current_df.fillna(0).values - self.train_mean) / self.train_std) * c_mask

            self.observed_mask.append(c_mask)
            self.gt_mask.append(c_gt_mask)
            self.observed_data.append(c_data)

            if mode == "test":
                n_sample = len(current_df) // eval_length
                c_index = np.arange(last_index, last_index + eval_length * n_sample, eval_length)
                self.use_index += c_index.tolist()
                self.cut_length += [0] * len(c_index)
                if len(current_df) % eval_length != 0:
                    self.use_index += [len(self.index_group) - 1]
                    self.cut_length += [eval_length - len(current_df) % eval_length]

        if mode != "test":
            self.use_index = np.arange(len(self.index_group))
            self.cut_length = [0] * len(self.use_index)

        if mode == "train":
            ind = -1
            self.index_group_histmask = []
            self.position_in_group_histmask = []

            for i in range(len(self.index_group)):
                while True:
                    ind += 1
                    if ind == len(self.index_group):
                        ind = 0
                    if self.valid_for_histmask[ind] == 1:
                        self.index_group_histmask.append(self.index_group[ind])
                        self.position_in_group_histmask.append(self.position_in_group[ind])
                        break
        else:
            self.index_group_histmask = self.index_group
            self.position_in_group_histmask = self.position_in_group

    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        c_group = self.index_group[index]
        c_index = self.position_in_group[index]
        hist_group = self.index_group_histmask[index]
        hist_index = self.position_in_group_histmask[index]
        s = {
            "observed_data": self.observed_data[c_group][c_index : c_index + self.eval_length],
            "observed_mask": self.observed_mask[c_group][c_index : c_index + self.eval_length],
            "gt_mask": self.gt_mask[c_group][c_index : c_index + self.eval_length],
            "hist_mask": self.observed_mask[hist_group][hist_index : hist_index + self.eval_length],
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
        }

        return s

    def __len__(self):
        return len(self.use_index)


def get_dataloader(batch_size, device, validindex=0):
    dataset = UserBehaviorDataset(mode="train", validindex=validindex)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    dataset_test = UserBehaviorDataset(mode="test", validindex=validindex)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=1, shuffle=False)
    dataset_valid = UserBehaviorDataset(mode="valid", validindex=validindex)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, num_workers=1, shuffle=False)

    scaler = torch.from_numpy(dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(dataset.train_mean).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler


# Описание изменений
# Атрибуты: Заменены атрибуты, связанные с показателями PM2.5, на атрибуты пользовательского поведения, такие как "likes", "comments", "time_spent".
# Источник данных: Данные подгружаются из файлов behavior_data.csv и behavior_gt.csv, которые представляют пользовательские действия и истинные значения.
# Режимы: Код поддерживает режимы "train", "valid", и "test" с разными наборами месяцев для каждого режима, позволяя отделить данные для обучения и тестирования.
# Теперь можно использовать этот код для обучения модели, такой как BRITS, на данных поведения пользователя.