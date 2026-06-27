# ======================================================
# HOTEL BOOKING CANCELLATION PREDICTION
# ======================================================

# Import Libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create folder to save graphs
os.makedirs("Graphs", exist_ok=True)

# ======================================================
# Load Dataset
# ======================================================

url = "https://raw.githubusercontent.com/Anu4477/DS-projects/main/hotel_bookings.csv"

df = pd.read_csv(url)

# Take only 5000 random rows
df = df.sample(n=5000, random_state=42)

print(df.head())
print("\nShape:", df.shape)

# ======================================================
# Data Cleaning
# ======================================================

# Remove unwanted index column
df.drop("index", axis=1, inplace=True)

# Missing Values
df["children"] = df["children"].fillna(0)
df["country"] = df["country"].fillna(df["country"].mode()[0])
df["agent"] = df["agent"].fillna(0)
df["company"] = df["company"].fillna(0)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Remove unnecessary columns
df.drop(
    ["reservation_status", "reservation_status_date"],
    axis=1,
    inplace=True
)

print(df.isnull().sum())

print("\nNew Shape:", df.shape)

# ======================================================
# Exploratory Data Analysis (EDA)
# ======================================================

sns.set_theme(style="whitegrid", font_scale=1.1)

plt.figure(figsize=(6,4))

sns.countplot(
    data=df,
    x="is_canceled",
    hue="is_canceled",
    palette="Set2",
    legend=False
)

plt.title("Booking Cancellation Distribution")
plt.xlabel("Cancelled (0 = No, 1 = Yes)")
plt.ylabel("Number of Bookings")

plt.savefig("Graphs/booking_cancellation.png",
            dpi=300,
            bbox_inches="tight")

plt.show()

plt.figure(figsize=(6,4))

sns.countplot(
    data=df,
    x="hotel",
    hue="hotel",
    palette="pastel",
    legend=False
)

plt.title("Hotel Type Distribution")
plt.xlabel("Hotel Type")
plt.ylabel("Count")

plt.savefig("Graphs/hotel_type.png",
            dpi=300,
            bbox_inches="tight")

plt.show()

plt.figure(figsize=(6,4))

sns.countplot(
    data=df,
    x="hotel",
    hue="hotel",
    palette="pastel",
    legend=False
)

plt.title("Hotel Type Distribution")
plt.xlabel("Hotel Type")
plt.ylabel("Count")

plt.savefig("Graphs/hotel_type.png",
            dpi=300,
            bbox_inches="tight")

plt.show()

plt.figure(figsize=(10,5))

order = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

sns.countplot(
    data=df,
    x="arrival_date_month",
    order=order,
    color="steelblue"
)

plt.xticks(rotation=45)

plt.title("Monthly Hotel Bookings")
plt.xlabel("Month")
plt.ylabel("Bookings")

plt.savefig("Graphs/monthly_bookings.png",
            dpi=300,
            bbox_inches="tight")

plt.show()


plt.figure(figsize=(8,5))

sns.histplot(
    df["lead_time"],
    bins=30,
    kde=True,
    color="purple"
)

plt.title("Lead Time Distribution")
plt.xlabel("Lead Time")
plt.ylabel("Frequency")

plt.savefig("Graphs/lead_time.png",
            dpi=300,
            bbox_inches="tight")

plt.show()

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="is_canceled",
    y="adr",
    hue="is_canceled",
    palette="coolwarm",
    legend=False
)

plt.title("Average Daily Rate vs Cancellation")
plt.xlabel("Cancelled")
plt.ylabel("ADR")

plt.savefig("Graphs/adr_boxplot.png",
            dpi=300,
            bbox_inches="tight")

plt.show()

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="deposit_type",
    hue="deposit_type",
    palette="viridis",
    legend=False
)

plt.title("Deposit Type Distribution")
plt.xticks(rotation=20)

plt.savefig("Graphs/deposit_type.png",
            dpi=300,
            bbox_inches="tight")

plt.show()

plt.figure(figsize=(10,6))

sns.countplot(
    data=df,
    y="market_segment",
    order=df["market_segment"].value_counts().index,
    color="teal"
)

plt.title("Market Segment Distribution")
plt.xlabel("Bookings")
plt.ylabel("Market Segment")

plt.savefig("Graphs/market_segment.png",
            dpi=300,
            bbox_inches="tight")

plt.show()

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="RdBu_r",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.savefig("Graphs/correlation_heatmap.png",
            dpi=300,
            bbox_inches="tight")

plt.show()

# ======================================================
# Encoding Categorical Columns
# ======================================================

encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

print(df.head())

# ======================================================
# Features & Target
# ======================================================

X = df.drop("is_canceled", axis=1)
y = df["is_canceled"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# ======================================================
# Train-Test Split
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data :", X_test.shape)

# ======================================================
# Feature Scaling
# ======================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================================================
# Logistic Regression Model
# ======================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully!")


# ======================================================
# Model Accuracy
# ======================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2%}")

# ======================================================
# Predict New Booking
# ======================================================

sample = X.iloc[[0]].copy()

# Change a few values
sample["lead_time"] = 120
sample["adults"] = 2
sample["children"] = 1
sample["adr"] = 180
sample["total_of_special_requests"] = 2

# Scale sample
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)

print("\nPrediction Result")

if prediction[0] == 1:
    print("❌ Booking will be Cancelled")
else:
    print("✅ Booking will NOT be Cancelled")

    # ======================================================
# Save Model
# ======================================================

joblib.dump(model, "hotel_booking_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model Saved Successfully!")