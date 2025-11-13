import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 60000

days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
time_of_booking = ["Morning", "Afternoon", "Evening", "Night"]
train_popularity = ["High", "Medium", "Low"]
season = ["Holiday", "Festival", "Normal"]
travel_class = ["Sleeper", "3AC", "2AC", "1AC"]
booking_type = ["Normal", "Tatkal"]

data = {
    "Day_of_Week": np.random.choice(days_of_week, n_samples),
    "Time_of_Booking": np.random.choice(time_of_booking, n_samples),
    "Days_Before_Travel": np.random.randint(0, 121, n_samples),
    "Train_Popularity": np.random.choice(train_popularity, n_samples),
    "Season": np.random.choice(season, n_samples),
    "Travel_Class": np.random.choice(travel_class, n_samples),
    "Num_Passengers": np.random.randint(1, 6, n_samples),
    "Booking_Type": np.random.choice(booking_type, n_samples),
    "Waiting_List_Position": np.random.randint(0, 101, n_samples),
}

df = pd.DataFrame(data)

def ticket_confirmed(row):
    prob = 0.5
    if row["Days_Before_Travel"] > 60:
        prob += 0.2
    if row["Train_Popularity"] == "High":
        prob -= 0.2
    if row["Season"] == "Festival":
        prob -= 0.15
    if row["Booking_Type"] == "Tatkal":
        prob -= 0.1
    if row["Waiting_List_Position"] < 10:
        prob += 0.3
    if row["Travel_Class"] in ["2AC", "1AC"]:
        prob += 0.1

    return np.random.rand() < prob

df["Ticket_Confirmed"] = df.apply(ticket_confirmed, axis=1).astype(int)
df.to_csv("train_ticket_booking_dataset_50000.csv", index=False)
