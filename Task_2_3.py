import pandas as pd
import json

# Load CSV
df = pd.read_csv("rasa_logs.csv")

# Extract user messages
def extract_user_data(row):
    try:
        d = json.loads(row)
        if d.get("event") == "user":
            return pd.Series([d.get("event"), d.get("text", "")])
        else:
            return pd.Series([None, None])
    except:
        return pd.Series([None, None])

df[["event_type", "text"]] = df["data"].apply(extract_user_data)
user_df = df[df["event_type"] == "user"][["sender_id", "timestamp", "intent_name", "text"]].copy()
user_df.dropna(subset=["text"], inplace=True)

print("Sample user messages:")
print(user_df.head())

# -------------------------
# Conversion rate per user
# -------------------------
user_summary = (
    user_df.groupby("sender_id")
    .agg(total_messages=("text", "count"),
         fallback_messages=("intent_name", lambda x: (x == "nlu_fallback").sum()))
    .reset_index()
)

user_summary["conversion_rate"] = 1 - (user_summary["fallback_messages"] / user_summary["total_messages"])

print("\nUser-level conversion rates:")
print(user_summary[["sender_id", "conversion_rate"]])

overall_conversion_rate = user_summary["conversion_rate"].mean()
print("\nOverall Conversion Rate:", round(overall_conversion_rate*100, 2), "%")

# -------------------------
# CSAT (per-message basis)
# -------------------------
csat_score = 1 - (user_df["intent_name"] == "nlu_fallback").mean()
print("Estimated CSAT:", round(csat_score*100, 2), "%")

# -------------------------
# NPS classification
# -------------------------
def classify_nps(ratio):
    if ratio < 0.2:
        return "Promoter"
    elif ratio < 0.5:
        return "Passive"
    else:
        return "Detractor"

user_summary["fallback_ratio"] = user_summary["fallback_messages"] / user_summary["total_messages"]
user_summary["nps_group"] = user_summary["fallback_ratio"].apply(classify_nps)

promoters = (user_summary["nps_group"] == "Promoter").sum()
detractors = (user_summary["nps_group"] == "Detractor").sum()
total_users = len(user_summary)
nps_score = ((promoters - detractors) / total_users) * 100

print("\nEstimated NPS Score:", round(nps_score, 2))
