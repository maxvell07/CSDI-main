import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from datetime import datetime

# Список событий, которые могут быть в пользовательских данных
event_types = ["page_view", "click", "form_submit", "scroll", "hover"]

def parse_data(x):
    # Преобразуем тип события в числовой формат
    x = x.set_index("event_type").to_dict()["duration"]
    
    values = []
    for event in event_types:
        if x.__contains__(event):
            values.append(x[event])
        else:
            values.append(np.nan)
    return values

def parse_user(user_id, missing_ratio=0.1):
    data = pd.read_csv(f"./data/user_behavior/{user_id}.txt", sep="\t")
    
    # Обработка временной метки
    data["timestamp"] = data["timestamp"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)
    
    observed_values = []
    for hour in range(24):
        observed_values.append(parse_data(data[data["timestamp"] == hour]))
    observed_values = np.array(observed_values)
    observed_masks = ~np.isnan(observed_values)

    # Случайное задание некоторого процента отсутствующих данных
    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    miss_indices = np.random.choice(
        obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
    )
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks

def get_userlist():
    user_list = []
    for filename in os.listdir("./data/user_behavior"):
        match = re.search("\d+", filename)
        if match:
            user_list.append(match.group())
    user_list = np.sort(user_list)
    return user_list

class UserBehavior_Dataset(Dataset):
    def __init__(self, eval_length=24, use_index_list=None, missing_ratio=0.0, seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        path = f"./data/user_behavior_missing{missing_ratio}_seed{seed}.pk"

        if not os.path.isfile(path):
            user_list = get_userlist()
            for user_id in user_list:
                try:
                    observed_values, observed_masks, gt_masks = parse_user(user_id, missing_ratio)
                    self.observed_values.append(observed_values)
                    self.observed_masks.append(observed_masks)
                    self.gt_masks.append(gt_masks)
                except Exception as e:
                    print(user_id, e)
                    continue
            self.observed_values = np.array(self.observed_values)
            self.observed_masks = np.array(self.observed_masks)
            self.gt_masks = np.array(self.gt_masks)

            # Нормализация данных
            tmp_values = self.observed_values.reshape(-1, len(event_types))
            tmp_masks = self.observed_masks.reshape(-1, len(event_types))
            mean = np.zeros(len(event_types))
            std = np.zeros(len(event_types))
            for k in range(len(event_types)):
                c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
                mean[k] = c_data.mean()
                std[k] = c_data.std()
            self.observed_values = ((self.observed_values - mean) / std) * self.observed_masks

            with open(path, "wb") as f:
                pickle.dump([self.observed_values, self.observed_masks, self.gt_masks], f)
        else:
            with open(path, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(f)

        self.use_index_list = np.arange(len(self.observed_values)) if use_index_list is None else use_index_list

    def __getitem__(self, index):
        index = self.use_index_list[index]
        return {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }

    def __len__(self):
        return len(self.use_index_list)

def get_user_behavior_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1):
    dataset = UserBehavior_Dataset(missing_ratio=missing_ratio, seed=seed)
    indlist = np.arange(len(dataset))

    np.random.seed(seed)
    np.random.shuffle(indlist)

    start = int(nfold * 0.2 * len(dataset)) if nfold else 0
    end = int((nfold + 1) * 0.2 * len(dataset)) if nfold else len(dataset)
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = int(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    train_loader = DataLoader(UserBehavior_Dataset(use_index_list=train_index, missing_ratio=missing_ratio, seed=seed), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(UserBehavior_Dataset(use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(UserBehavior_Dataset(use_index_list=test_index, missing_ratio=missing_ratio, seed=seed), batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader
