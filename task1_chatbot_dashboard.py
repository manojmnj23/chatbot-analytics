# Import Lib
import pandas as pd

# Dataset Load
df = pd.read_csv("rasa_logs.csv")
print(df.head())

# Data Cleaning

user_df = df[df["intent_name"].notna()]
user_df = user_df[["sender_id", "timestamp", "intent_name"]]
user_df.reset_index(drop=True, inplace=True)

print(user_df.head())


# bot messages

bot_df = df[df["action_name"].notna()]
bot_df = bot_df[["sender_id", "timestamp", "action_name"]]

print(bot_df.head())


# USER SEGMENTATION BY INTENT FREQUENCY

intent_by_user = (
    user_df
    .groupby(["sender_id", "intent_name"])
    .size()
    .reset_index(name="count")
)

intent_by_user.head()


# Convert to wide format

intent_pivot = intent_by_user.pivot(
    index="sender_id",
    columns="intent_name",
    values="count"
).fillna(0)

intent_pivot.head()


# VISUALISATION 1 – Intent Distribution (Dashboard)


import matplotlib.pyplot as plt

intent_totals = user_df["intent_name"].value_counts()

plt.figure(figsize=(10,5))
intent_totals.plot(kind="bar")
plt.title("Overall Intent Distribution in E-commerce Chatbot")
plt.xlabel("Intent")
plt.ylabel("Number of User Queries")
plt.show()


# FALLBACK RATE ANALYSIS (CRITICAL INSIGHT)

# TOTAL USER QUERIES
total_queries = len(user_df)

# FALLBACK ACTIONS (bot-side)
fallback_actions = bot_df[bot_df["action_name"] == "action_default_fallback"]

# FALLBACK RATE
fallback_rate = len(fallback_actions) / total_queries if total_queries > 0 else 0
print(f"Fallback rate: {round(fallback_rate * 100, 2)}%")
print("Total user queries:", total_queries)
print("Fallback queries:", fallback_actions)


fallback_by_user = (
    fallback_actions
    .groupby("sender_id")
    .size()
    .reset_index(name="fallback_count")
    .sort_values("fallback_count", ascending=False)
)

print(fallback_by_user.head())

print(bot_df["action_name"].value_counts().head(10))

# TABLE – USER vs INTENT FREQUENCY
user_intent_table = (
    user_df
    .groupby(["sender_id", "intent_name"])
    .size()
    .reset_index(name="count")
)

user_intent_table.head(10)

user_intent_pivot = user_intent_table.pivot(
    index="sender_id",
    columns="intent_name",
    values="count"
).fillna(0)

user_intent_pivot.head()


intent_totals.to_csv("intent_distribution.csv")
user_intent_pivot.to_csv("user_intent_frequency.csv")




