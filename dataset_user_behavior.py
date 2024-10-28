import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import pickle

# Список событий, которые могут быть в пользовательских данных
event_types = ["page_view", "click", "form_submit", "scroll", "hover"]

def parse_data(x):
    x = x.set_index("event_type").to_dict()["duration"]
    values = []
    for event in event_types:
        values.append(x.get(event, np.nan))  # Используйте get для безопасного получения значений
    return values

def parse_user(user_id, missing_ratio=0.1):
    data = pd.read_csv("./user_behavior_data.txt", sep="\t")
    
    # Фильтруем данные для конкретного пользователя
    user_data = data[data['user_id'] == user_id]
    
    # Проверка на наличие данных для конкретного пользователя
    if user_data.empty:
        raise ValueError(f"No data found for user_id: {user_id}")
    
    # Обработка временной метки
    user_data.loc[:, "timestamp"] = user_data["timestamp"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").hour)
    
    observed_values = []
    observed_masks = []  # Инициализация переменной для масок

    for hour in range(24):
        hour_data = user_data[user_data["timestamp"] == hour]
        parsed_data = parse_data(hour_data)
        observed_values.append(parsed_data)

        # Добавление масок для текущего часа
        masks = ~np.isnan(parsed_data)
        observed_masks.append(masks)

    observed_values = np.array(observed_values)
    observed_masks = np.array(observed_masks)

    # Проверка на наличие наблюдаемых значений
    if observed_values.size == 0:
        raise ValueError(f"No observed values for user_id: {user_id}")

    # Случайное задание некоторого процента отсутствующих данных
    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    
    if len(obs_indices) == 0:
        raise ValueError(f"No observed indices found for user_id: {user_id}")

    miss_indices = np.random.choice(
        obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
    )
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


class UserBehavior_Dataset(Dataset):
    def __init__(self, eval_length=24, use_index_list=None, missing_ratio=0.0, seed=0):
        self.eval_length = eval_length
        np.random.seed(seed)

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        path = f"./data/user_behavior_missing{missing_ratio}_seed{seed}.pk"

        print ("test")
        if not os.path.isfile(path):
            print ("test")
            # Загрузка данных из одного файла без использования user_id
            for user_id in range(1, 1001):  # Измените это значение на количество пользователей
                observed_values, observed_masks, gt_masks = parse_user(user_id, missing_ratio)
                self.observed_values.append(observed_values)
                self.observed_masks.append(observed_masks)
                self.gt_masks.append(gt_masks)

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
                if c_data.size > 0:  # Добавьте проверку на наличие данных
                    mean[k] = c_data.mean()
                    std[k] = c_data.std() if c_data.std() != 0 else 1  # Избегайте деления на ноль
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
    print(f"Initial dataset size: {len(indlist)}")  # Вывод размера датасета

    # Проверка на наличие данных в датасете
    if len(indlist) == 0:
        raise ValueError("Dataset is empty. Ensure data is loaded correctly.")

    np.random.seed(seed)
    np.random.shuffle(indlist)

    start = int(nfold * 0.2 * len(dataset)) if nfold else 0
    end = int((nfold + 1) * 0.2 * len(dataset)) if nfold else len(dataset)
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = int(len(dataset) * 0.7)
    
    print(f"Number of training samples before splitting: {num_train}")

    if num_train <= 0:
        raise ValueError("No training samples available.")

    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]

    print(f"Train index size: {len(train_index)}")
    print(f"Valid index size: {len(valid_index)}")
    print(f"Test index size: {len(test_index)}")

    train_loader = DataLoader(UserBehavior_Dataset(use_index_list=train_index, missing_ratio=missing_ratio, seed=seed), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(UserBehavior_Dataset(use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(UserBehavior_Dataset(use_index_list=test_index, missing_ratio=missing_ratio, seed=seed), batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

