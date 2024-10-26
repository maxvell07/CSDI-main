import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import pickle

# Атрибуты действий пользователей в соцсетях/на сайте банка
attributes = ['Likes', 'Comments', 'Shares', 'TimeSpent', 'PageViews', 'Clicks', 'ScrollDepth']

# Генерация данных на основе временных шагов (например, по часам)
def parse_user_data(time_step):
    data = {}
    for attr in attributes:
        if attr == 'TimeSpent':
            data[attr] = np.abs(np.random.normal(5, 2))  # Время на странице в минутах
        elif attr == 'PageViews':
            data[attr] = np.random.poisson(1)
        elif attr == 'ScrollDepth':
            data[attr] = np.random.uniform(0, 1)  # Доля страницы
        else:
            data[attr] = np.random.poisson(1)  # Другие действия
    return [data[attr] for attr in attributes]

def parse_user(id_, steps=48, missing_ratio=0.1):
    observed_values = [parse_user_data(t) for t in range(steps)]
    observed_values = np.array(observed_values)
    observed_masks = ~np.isnan(observed_values)

    # Установка пропусков для иммутации
    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    miss_indices = np.random.choice(obs_indices, int(len(obs_indices) * missing_ratio), replace=False)
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")
    
    return observed_values, observed_masks, gt_masks

class UserBehaviorDataset(Dataset):
    def __init__(self, eval_length=48, use_index_list=None, missing_ratio=0.1, seed=0):
        np.random.seed(seed)
        self.eval_length = eval_length
        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []

        user_ids = range(100)  # генерируем для 100 пользователей
        for user_id in user_ids:
            observed_values, observed_masks, gt_masks = parse_user(user_id, missing_ratio=missing_ratio)
            self.observed_values.append(observed_values)
            self.observed_masks.append(observed_masks)
            self.gt_masks.append(gt_masks)

        self.observed_values = np.array(self.observed_values)
        self.observed_masks = np.array(self.observed_masks)
        self.gt_masks = np.array(self.gt_masks)

    def __getitem__(self, index):
        return {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }

    def __len__(self):
        return len(self.observed_values)

def get_dataloader(batch_size=16, missing_ratio=0.1):
    dataset = UserBehaviorDataset(missing_ratio=missing_ratio)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Инициализация DataLoader
dataloader = get_dataloader()


# Описание кода
# Атрибуты: Заданы атрибуты для анализа действий пользователя, такие как "Likes", "Comments", "TimeSpent" и т.д.
# Генерация данных: parse_user_data генерирует данные для каждого временного шага, имитируя действия пользователя.
# Пропуски данных: Устанавливаются случайные пропуски в данных.
# DataLoader: Класс UserBehaviorDataset инициализирует набор данных, а get_dataloader создает DataLoader для использования в модели BRITS.
# Этот код можно использовать для обучения модели BRITS на синтетических данных, моделирующих поведение пользователей на сайте банка или в социальной сети.