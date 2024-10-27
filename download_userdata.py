import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Генерация данных
def generate_user_behavior_data(num_users=1000, num_sessions=50, start_date="2024-01-01"):
    random.seed(42)
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    data = []

    for user_id in range(1, num_users + 1):
        for session in range(1, num_sessions + 1):
            session_start = start_date + timedelta(days=random.randint(0, 365))
            num_events = random.randint(5, 20)
            session_duration = timedelta(minutes=random.randint(10, 120))

            # Записываем события в рамках сессии
            for event in range(num_events):
                timestamp = session_start + timedelta(seconds=random.randint(0, int(session_duration.total_seconds())))
                event_type = random.choice(["page_view", "click", "form_submit", "scroll", "hover"])
                duration = random.randint(1, 300) if event_type == "page_view" else 0

                data.append({
                    "user_id": user_id,
                    "session_id": session,
                    "timestamp": timestamp,
                    "event_type": event_type,
                    "duration": duration
                })

    return pd.DataFrame(data)

# Пример генерации данных
user_behavior_data = generate_user_behavior_data()
user_behavior_data = user_behavior_data.sort_values(by=["user_id", "session_id", "timestamp"])
print(user_behavior_data.head(10))

# Сохранение данных в .txt файл
def save_data_to_txt(df, filename="user_behavior_data.txt"):
    with open(filename, "w") as f:
        f.write("user_id\tsession_id\ttimestamp\tevent_type\tduration\n")  # Заголовок
        for index, row in df.iterrows():
            f.write(f"{row['user_id']}\t{row['session_id']}\t{row['timestamp']}\t{row['event_type']}\t{row['duration']}\n")

# Сохранение сгенерированных данных в текстовый файл
save_data_to_txt(user_behavior_data)
print(f"Данные сохранены в файл user_behavior_data.txt")
