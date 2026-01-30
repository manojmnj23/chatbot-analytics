import sqlite3
import pandas as pd

# connect to rasa database
conn = sqlite3.connect("rasa.db")

# read events table
df = pd.read_sql("SELECT * FROM events", conn)

# show first 5 rows
print(df.head())


df.to_csv("rasa_logs.csv", index=False)
