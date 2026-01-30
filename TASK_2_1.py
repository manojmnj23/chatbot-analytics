import pandas as pd
import json

# Load CSV
df = pd.read_csv("rasa_logs.csv")

# Keep only rows where intent_name exists and event is user
def extract_user_data(row):
    try:
        d = json.loads(row)
        # Only consider user messages
        if d.get("event") == "user":
            text = d.get("text", "")
            intent = d.get("parse_data", {}).get("intent", {}).get("name")
            return pd.Series([text, intent])
        else:
            return pd.Series([None, None])
    except:
        return pd.Series([None, None])

# Apply extraction
df[["text_extracted", "intent_extracted"]] = df["data"].apply(extract_user_data)

# Keep only valid user messages
user_df = df[df["text_extracted"].notna()][["sender_id", "timestamp", "intent_name", "text_extracted"]].copy()
user_df.rename(columns={"text_extracted": "text"}, inplace=True)

# Now define your true intent function
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

# Apply true intent
user_df["true_intent"] = user_df["text"].apply(assign_true_intent)
user_df[["text", "true_intent", "intent_name"]].head(10)
# Check first few rows
print(user_df.head(10))

from sklearn.metrics import accuracy_score
# Fallback rate
accuracy = accuracy_score(user_df["true_intent"], user_df["intent_name"])
print(f"Intent Recognition Accuracy: {round(accuracy*100, 2)}%")





#CONFUSION MATRIX

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

labels = sorted(user_df["true_intent"].unique())

cm = confusion_matrix(
    user_df["true_intent"],
    user_df["intent_name"],
    labels=labels
)
#Plot confusion matrix

plt.figure(figsize=(10, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels,
    cmap="Blues"
)
plt.xlabel("Predicted Intent")
plt.ylabel("True Intent")
plt.title("Confusion Matrix – E-commerce Chatbot Intent Classification")
plt.show()



# CLASSIFICATION REPORT
from sklearn.metrics import classification_report

print(classification_report(
    user_df["true_intent"],
    user_df["intent_name"]
))




