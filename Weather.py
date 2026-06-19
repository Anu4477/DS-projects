# Temperature Analysis Project

import pandas as pd
import matplotlib.pyplot as plt

# =========================
# STEP 1: Load Dataset
# =========================

url = "https://raw.githubusercontent.com/Anu4477/Data-science/main/temperature_data.json"
df = pd.read_json(url)

print("Dataset Loaded Successfully\n")

# =========================
# STEP 2: Data Exploration
# =========================

print("First 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

# =========================
# STEP 3: Data Cleaning
# =========================

# Remove rows where temperature is missing
df.dropna(subset=["temperature_c"], inplace=True)

# Fill missing humidity values with mean
df['humidity_pct'] = df['humidity_pct'].fillna(
    df['humidity_pct'].mean()
)

print("\nAfter Cleaning")
print(df.isnull().sum())

# =========================
# STEP 4: Feature Engineering
# =========================

# Celsius to Fahrenheit
df['temperature_f'] = (df['temperature_c'] * 1.8) + 32

print("\nDataset with Fahrenheit Column")
print(df.head())

# =========================
# STEP 5: Basic Analysis
# =========================

print("\nAverage Temperature:")
print(df['temperature_c'].mean())

print("\nMaximum Temperature:")
print(df['temperature_c'].max())

print("\nMinimum Temperature:")
print(df['temperature_c'].min())

print("\nAverage Humidity:")
print(df['humidity_pct'].mean())

print("\nHottest Day:")
print(df.loc[df['temperature_c'].idxmax(), 'day'])

print("\nMost Humid Day:")
print(df.loc[df['humidity_pct'].idxmax(), 'day'])

# =========================
# STEP 6: Visualization
# =========================

plt.style.use('ggplot')

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# -------------------------
# Pie Chart
# -------------------------

temp_colors = ['#FF9999','#66B3FF','#99FF99','#FFCC99','#C2C2F0','#FFD700']

ax[0].pie(
    df['temperature_c'],
    labels=df['day'],
    autopct='%1.1f%%',
    shadow=True,
    startangle=90,
    colors=temp_colors[:len(df)],
    wedgeprops={'edgecolor':'black'}
)

ax[0].set_title(
    "Temperature Distribution",
    fontsize=14,
    fontweight='bold'
)

# -------------------------
# Bar Chart
# -------------------------

colors = ['#FF6B6B','#4ECDC4','#45B7D1','#FFA07A','#98D8C8','#C7CEEA']

bars = ax[1].bar(
    df['day'],
    df['humidity_pct'],
    color=colors[:len(df)],
    edgecolor='black',
    linewidth=1.5
)

# Value Labels
for bar in bars:
    height = bar.get_height()
    ax[1].text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f'{height:.0f}%',
        ha='center',
        fontweight='bold'
    )

ax[1].set_title(
    'Humidity Levels by Day',
    fontsize=14,
    fontweight='bold'
)

ax[1].set_xlabel(
    'Day',
    fontsize=12,
    fontweight='bold'
)

ax[1].set_ylabel(
    'Humidity (%)',
    fontsize=12,
    fontweight='bold'
)

ax[1].grid(
    axis='y',
    linestyle='--',
    alpha=0.7
)

# Dashboard Title
plt.suptitle(
    'Weather Analysis Dashboard',
    fontsize=18,
    fontweight='bold'
)

plt.tight_layout()

plt.savefig(
    'weather_dashboard.jpg',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# =========================
# STEP 7: Final Dataset
# =========================

print("\nFinal Dataset")
print(df)