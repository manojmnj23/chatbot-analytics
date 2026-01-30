import pandas as pd
import json
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -------------------------------
# LOAD CSV
# -------------------------------
df = pd.read_csv("rasa_logs.csv")

# -------------------------------
# EXTRACT USER MESSAGES
# -------------------------------
def extract_user_data(row):
    """
    Extract 'event_type' and 'text' from JSON string in 'data'.
    Only returns text if event is 'user'.
    """
    try:
        d = json.loads(row)
        if d.get("event") == "user":
            text = d.get("text", "")
            return pd.Series([d.get("event"), text])
        else:
            return pd.Series([None, None])
    except:
        return pd.Series([None, None])

# Apply extraction
df[["event_type", "text"]] = df["data"].apply(extract_user_data)

# Keep only user messages
user_df = df[df["event_type"] == "user"][["sender_id", "timestamp", "intent_name", "text"]].copy()
user_df.dropna(subset=["text"], inplace=True)

print("Sample user messages:")
print(user_df.head())

# -------------------------------
# BASIC DESCRIPTIVE STATISTICS
# -------------------------------
num_users = user_df["sender_id"].nunique()
num_queries = len(user_df)
num_intents = user_df["intent_name"].nunique()

print("\nNumber of unique users:", num_users)
print("Total user queries:", num_queries)
print("Number of unique intents:", num_intents)

# -------------------------------
# INTENT FREQUENCY DISTRIBUTION
# -------------------------------
intent_freq = user_df["intent_name"].value_counts()
print("\nIntent frequency distribution:")
print(intent_freq)

plt.figure(figsize=(10,5))
intent_freq.plot(kind="bar")
plt.title("Intent Frequency Distribution")
plt.xlabel("Intent")
plt.ylabel("Number of Queries")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# TRUE INTENT ASSIGNMENT
# -------------------------------
def assign_true_intent(text):
    text = str(text).lower()
    if "order" in text or "track" in text or "status" in text:
        return "order_status"
    elif "shoe" in text or "jacket" in text or "buy" in text:
        return "product_search"
    elif "ship" in text or "delivery" in text:
        return "shipping_info"
    elif "return" in text or "refund" in text:
        return "return_request"
    elif "hello" in text or "hi" in text:
        return "greet"
    elif "bye" in text:
        return "goodbye"
    else:
        return "nlu_fallback"

user_df["true_intent"] = user_df["text"].apply(assign_true_intent)

# -------------------------------
# FALLBACK ANALYSIS
# -------------------------------
fallback_by_user = (
    user_df[user_df["true_intent"] == "nlu_fallback"]
    .groupby("sender_id")
    .size()
    .reset_index(name="fallback_count")
)

queries_by_user = (
    user_df.groupby("sender_id")
    .size()
    .reset_index(name="total_queries")
)

user_fallback_analysis = pd.merge(
    queries_by_user,
    fallback_by_user,
    on="sender_id",
    how="left"
).fillna(0)

user_fallback_analysis["fallback_ratio"] = (
    user_fallback_analysis["fallback_count"] /
    user_fallback_analysis["total_queries"]
)

print("\nFallback count by user:")
print(fallback_by_user)

print("\nUser-level fallback analysis:")
print(user_fallback_analysis)

overall_fallback_rate = (user_df["true_intent"] == "nlu_fallback").mean()
print(f"\nOverall fallback rate: {round(overall_fallback_rate*100, 2)}%")

# -------------------------------
# INTENT COMPARISON TABLE
# -------------------------------
intent_comparison = user_df.groupby(["true_intent", "intent_name"]).size().reset_index(name="count")
print("\nIntent comparison table:")
print(intent_comparison.head(10))

# -------------------------------
# INTENT RECOGNITION ACCURACY
# -------------------------------
accuracy = accuracy_score(user_df["true_intent"], user_df["intent_name"])
print(f"\nIntent Recognition Accuracy: {round(accuracy*100, 2)}%")

# -------------------------------
# SESSION LENGTH ANALYSIS
# -------------------------------
session_length = (
    user_df.groupby("sender_id")
    .size()
    .reset_index(name="messages_per_session")
)

print("\nSession length stats:")
print(session_length.describe())

plt.figure(figsize=(8,5))
plt.hist(session_length["messages_per_session"], bins=10, color="skyblue", edgecolor="black")
plt.title("Session Length Distribution")
plt.xlabel("Messages per Session")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.show()

# -------------------------------
# RESPONSE SUCCESS VS FAILURE SPLIT
# -------------------------------
success_queries = user_df[user_df["intent_name"] != "nlu_fallback"]
failure_queries = user_df[user_df["intent_name"] == "nlu_fallback"]

print("\nSuccessful queries:", len(success_queries))
print("Fallback queries:", len(failure_queries))
