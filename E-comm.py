
# ==========================================================
# RECRUITER-LEVEL E-COMMERCE DASHBOARD (MATPLOTLIB ONLY)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# STYLE
# ==========================================================

plt.style.use("seaborn-v0_8-whitegrid")

# ==========================================================
# LOAD DATASET
# ==========================================================

url = "https://raw.githubusercontent.com/Anu4477/DS-projects/main/e-commerce.csv"

df = pd.read_csv(url)

# ==========================================================
# CLEANING
# ==========================================================

df["Description"] = df["Description"].fillna("Unknown")
df = df.dropna(subset=["CustomerID"])

df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

GBP_TO_INR = 115

df["Sales_INR"] = df["Quantity"] * df["UnitPrice"] * GBP_TO_INR
df["Profit_INR"] = df["Sales_INR"] * 0.20

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Month"] = df["InvoiceDate"].dt.strftime("%Y-%m")

# ==========================================================
# KPIs
# ==========================================================

total_revenue = df["Sales_INR"].sum()
total_profit = df["Profit_INR"].sum()
total_orders = df["InvoiceNo"].nunique()
total_customers = df["CustomerID"].nunique()
avg_order_value = total_revenue / total_orders

# ==========================================================
# ANALYSIS
# ==========================================================

monthly_revenue = df.groupby("Month")["Sales_INR"].sum().sort_index()
monthly_profit = df.groupby("Month")["Profit_INR"].sum().sort_index()

top_products = df.groupby("Description")["Sales_INR"].sum().sort_values(ascending=False).head(10)

top_customers = df.groupby("CustomerID")["Sales_INR"].sum().sort_values(ascending=False).head(10)

country_sales = df.groupby("Country")["Sales_INR"].sum().sort_values(ascending=False).head(10)

customer_value = df.groupby("CustomerID")["Sales_INR"].sum()

segments = pd.cut(customer_value, bins=[0, 50000, 300000, np.inf], labels=["Low", "Medium", "High"])
segment_counts = segments.value_counts()

# ==========================================================
# FIGURE
# ==========================================================

fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor("#f8f9fa")

gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.35)

fig.suptitle("E-COMMERCE ANALYTICS DASHBOARD (INR)", fontsize=22, fontweight="bold")

# ==========================================================
# KPI CARDS
# ==========================================================

fig.text(0.03, 0.92, f"Revenue\n₹{total_revenue:,.0f}",
         bbox=dict(facecolor="#d4edda", boxstyle="round"), fontsize=12, fontweight="bold")

fig.text(0.25, 0.92, f"Profit\n₹{total_profit:,.0f}",
         bbox=dict(facecolor="#d1ecf1", boxstyle="round"), fontsize=12, fontweight="bold")

fig.text(0.47, 0.92, f"Orders\n{total_orders:,}",
         bbox=dict(facecolor="#fff3cd", boxstyle="round"), fontsize=12, fontweight="bold")

fig.text(0.69, 0.92, f"Customers\n{total_customers:,}",
         bbox=dict(facecolor="#f8d7da", boxstyle="round"), fontsize=12, fontweight="bold")

# ==========================================================
# 1. MONTHLY REVENUE
# ==========================================================

ax1 = fig.add_subplot(gs[0,0])
ax1.plot(monthly_revenue.index, monthly_revenue.values, marker="o", color="#6C5CE7", linewidth=3)
ax1.set_title("Monthly Revenue")
ax1.tick_params(axis='x', rotation=45)

# ==========================================================
# 2. MONTHLY PROFIT
# ==========================================================

ax2 = fig.add_subplot(gs[0,1])
ax2.plot(monthly_profit.index, monthly_profit.values, marker="o", color="#00B894", linewidth=3)
ax2.set_title("Monthly Profit")
ax2.tick_params(axis='x', rotation=45)

# ==========================================================
# 3. CUSTOMER SEGMENTS
# ==========================================================

ax3 = fig.add_subplot(gs[0,2])
ax3.pie(segment_counts, labels=segment_counts.index, autopct="%1.1f%%")
ax3.set_title("Customer Segments")

# ==========================================================
# 4. TOP PRODUCTS
# ==========================================================

ax4 = fig.add_subplot(gs[1,0])
colors = plt.cm.Blues(np.linspace(0.4,1,len(top_products)))
ax4.barh(top_products.index, top_products.values, color=colors)
ax4.set_title("Top Products")

# ==========================================================
# 5. TOP CUSTOMERS
# ==========================================================

ax5 = fig.add_subplot(gs[1,1])
colors = plt.cm.Oranges(np.linspace(0.4,1,len(top_customers)))
ax5.barh(top_customers.index.astype(str), top_customers.values, color=colors)
ax5.set_title("Top Customers")

# ==========================================================
# 6. COUNTRY SALES
# ==========================================================

ax6 = fig.add_subplot(gs[1,2])
colors = plt.cm.viridis(np.linspace(0.4,1,len(country_sales)))
ax6.barh(country_sales.index, country_sales.values, color=colors)
ax6.set_title("Country Sales")

# ==========================================================
# 7. DISTRIBUTION
# ==========================================================

ax7 = fig.add_subplot(gs[2,0])
ax7.hist(df["Sales_INR"], bins=40, color="#00CEC9", edgecolor="black")
ax7.set_title("Sales Distribution")

# ==========================================================
# 8. SCATTER
# ==========================================================

ax8 = fig.add_subplot(gs[2,1])
ax8.scatter(df["Sales_INR"], df["Profit_INR"], alpha=0.4, color="#6C5CE7")
ax8.set_title("Revenue vs Profit")

# ==========================================================
# 9. SUMMARY
# ==========================================================

ax9 = fig.add_subplot(gs[2,2])
ax9.axis("off")

ax9.text(0,0.9,
f"""SUMMARY

Revenue: ₹{total_revenue:,.0f}
Profit: ₹{total_profit:,.0f}
Orders: {total_orders:,}
Customers: {total_customers:,}
Avg Order: ₹{avg_order_value:,.0f}
Top Country: {country_sales.index[0]}
Top Product: {top_products.index[0][:25]}
""")

# ==========================================================
# SAVE
# ==========================================================

plt.tight_layout(rect=[0,0.03,1,0.9])
plt.savefig("Recruiter_Level_Ecommerce_Dashboard.png", dpi=300)
plt.show()

print("Dashboard Created Successfully")
