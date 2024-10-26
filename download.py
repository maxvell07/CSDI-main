import os
import numpy as np
import pandas as pd
import pickle
import tarfile
import zipfile
import sys
import wget
import requests

os.makedirs("data/", exist_ok=True)

if sys.argv[1] == "physio":
    url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"
    wget.download(url, out="data")
    with tarfile.open("data/set-a.tar.gz", "r:gz") as t:
        t.extractall(path="data/physio")

elif sys.argv[1] == "pm25":
    url = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/STMVL-Release.zip"
    urlData = requests.get(url).content
    filename = "data/STMVL-Release.zip"
    with open(filename, mode="wb") as f:
        f.write(urlData)
    with zipfile.ZipFile(filename) as z:
        z.extractall("data/pm25")

elif sys.argv[1] == "user_behavior":
    # Генерация синтетических данных о поведении пользователей
    def generate_user_behavior_data(num_users=100, steps=48, missing_ratio=0.1):
        attributes = ['Likes', 'Comments', 'Shares', 'TimeSpent', 'PageViews', 'Clicks', 'ScrollDepth']
        user_data = []

        for user_id in range(num_users):
            observed_values = []
            for step in range(steps):
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
                observed_values.append(data)

            observed_values = np.array(observed_values)

            # Создание пропусков
            masks = np.random.rand(steps, len(attributes)) > missing_ratio
            gt_masks = masks.copy()
            observed_values[~masks] = np.nan  # Установка пропусков в данные

            user_data.append((observed_values, masks, gt_masks))

        return user_data

    # Сохранение данных в формате pickle
    def save_user_behavior_data(user_data):
        path = "./data/user_behavior.txt"
        with open(path, "wb") as f:
            pickle.dump(user_data, f)

    # Генерация и сохранение данных
    user_data = generate_user_behavior_data()
    save_user_behavior_data(user_data)

    print("Синтетические данные о поведении пользователей успешно сгенерированы и сохранены.")
