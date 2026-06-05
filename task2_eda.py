"""
CodeAlpha Internship — Task 2: Exploratory Data Analysis (EDA)
==============================================================
Dataset : books_scraped.csv  (produced by Task 1 web scraping)
Goal    : Perform thorough EDA — structure, statistics, distributions,
          correlations, trends, anomalies — and save publication-quality charts.

Run     : python task2_eda.py
Outputs : eda_report.txt  +  six PNG charts  (all in current directory)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE   = ["#4f46e5","#0891b2","#059669","#d97706","#dc2626",
             "#7c3aed","#db2777","#0f766e","#b45309","#6b7280"]
BG        = "#f8f7f4"
CARD      = "#ffffff"
TEXT      = "#1a1a1a"
MUTED     = "#6b7280"
ACCENT    = "#4f46e5"

def style():
    plt.rcParams.update({
        "figure.facecolor"   : BG,
        "axes.facecolor"     : CARD,
        "axes.edgecolor"     : "#e5e3de",
        "axes.labelcolor"    : TEXT,
        "axes.titleweight"   : "bold",
        "axes.titlesize"     : 13,
        "axes.labelsize"     : 11,
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "xtick.color"        : MUTED,
        "ytick.color"        : MUTED,
        "xtick.labelsize"    : 9,
        "ytick.labelsize"    : 9,
        "grid.color"         : "#ece9e3",
        "grid.linewidth"     : 0.6,
        "legend.frameon"     : False,
        "font.family"        : "sans-serif",
        "figure.dpi"         : 130,
    })

style()

CSV = "books_scraped.csv"
df  = pd.read_csv(CSV)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
report_lines = []
def log(line=""):
    report_lines.append(line)
    print(line)

log("=" * 65)
log("  CodeAlpha — Task 2: Exploratory Data Analysis (EDA)")
log("  Dataset : Books Scraped from books.toscrape.com")
log("=" * 65)

log("\n── 1. SHAPE & DATA TYPES ──────────────────────────────────────")
log(f"  Rows    : {df.shape[0]:,}")
log(f"  Columns : {df.shape[1]}")
log(f"\n{df.dtypes.to_string()}")

log("\n── 2. FIRST 5 ROWS ────────────────────────────────────────────")
log(df.head().to_string())

log("\n── 3. MISSING VALUES ──────────────────────────────────────────")
missing = df.isnull().sum()
log(missing[missing > 0].to_string() if missing.any() else "  No missing values found.")

log("\n── 4. DUPLICATE ROWS ──────────────────────────────────────────")
dupes = df.duplicated().sum()
log(f"  Duplicate rows: {dupes}")

log("\n── 5. DESCRIPTIVE STATISTICS ──────────────────────────────────")
log(df.describe(include="all").to_string())

log("\n── 6. CATEGORY DISTRIBUTION ───────────────────────────────────")
cat_counts = df["category"].value_counts()
log(cat_counts.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — UNIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
log("\n── 7. PRICE STATISTICS ────────────────────────────────────────")
p = df["price_gbp"]
log(f"  Min     : £{p.min():.2f}")
log(f"  Max     : £{p.max():.2f}")
log(f"  Mean    : £{p.mean():.2f}")
log(f"  Median  : £{p.median():.2f}")
log(f"  Std Dev : £{p.std():.2f}")
log(f"  Skew    : {p.skew():.3f}")
log(f"  Kurt    : {p.kurtosis():.3f}")

# IQR-based outlier detection
Q1, Q3 = p.quantile(0.25), p.quantile(0.75)
IQR = Q3 - Q1
outliers = df[(p < Q1 - 1.5*IQR) | (p > Q3 + 1.5*IQR)]
log(f"\n  IQR     : £{IQR:.2f}  (Q1={Q1:.2f}, Q3={Q3:.2f})")
log(f"  Outliers (price): {len(outliers)} books")

log("\n── 8. RATING DISTRIBUTION ─────────────────────────────────────")
rc = df["rating"].value_counts().sort_index()
for star, cnt in rc.items():
    bar = "█" * int(cnt / 10)
    log(f"  {star}★  {cnt:>4}  {bar}")

log("\n── 9. STOCK STATUS ────────────────────────────────────────────")
stock = df["in_stock"].value_counts()
log(f"  In Stock    : {stock.get(True, 0)}")
log(f"  Out of Stock: {stock.get(False, 0)}")
log(f"  In-Stock %  : {100*stock.get(True,0)/len(df):.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BIVARIATE & MULTIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
log("\n── 10. PRICE BY RATING (mean) ─────────────────────────────────")
log(df.groupby("rating")["price_gbp"].mean().round(2).to_string())

log("\n── 11. CORRELATION MATRIX (numeric columns) ───────────────────")
num_cols = ["price_gbp","rating","pages","year_published","num_reviews"]
log(df[num_cols].corr().round(3).to_string())

log("\n── 12. TOP 10 CATEGORIES BY AVERAGE PRICE ─────────────────────")
log(df.groupby("category")["price_gbp"].mean().sort_values(ascending=False).head(10).round(2).to_string())

log("\n── 13. STATISTICAL TEST: Price vs Rating (Pearson r) ──────────")
r, pval = stats.pearsonr(df["price_gbp"], df["rating"])
log(f"  r = {r:.4f},  p-value = {pval:.4f}")
log(f"  {'Significant' if pval < 0.05 else 'Not significant'} correlation (α=0.05)")

log("\n── 14. HYPOTHESES & FINDINGS ──────────────────────────────────")
log("  H1: Higher-rated books cost more → " +
    ("Supported" if r > 0.05 and pval < 0.05 else "Not supported"))
in_stock_rating = df.groupby("in_stock")["rating"].mean()
diff = in_stock_rating[True] - in_stock_rating[False]
log(f"  H2: In-stock books have higher ratings →"
    f" Avg diff = {diff:+.3f} ({'Supported' if diff > 0 else 'Not supported'})")

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Chart 1: Price Distribution ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Price Distribution — Books Dataset", fontsize=14, fontweight="bold", color=TEXT, y=1.01)

ax = axes[0]
ax.hist(df["price_gbp"], bins=40, color=ACCENT, edgecolor=BG, linewidth=0.4)
ax.axvline(df["price_gbp"].mean(),   color="#dc2626", linewidth=1.6, linestyle="--", label=f"Mean £{df['price_gbp'].mean():.2f}")
ax.axvline(df["price_gbp"].median(), color="#059669", linewidth=1.6, linestyle=":",  label=f"Median £{df['price_gbp'].median():.2f}")
ax.set_xlabel("Price (£)")
ax.set_ylabel("Number of Books")
ax.set_title("Histogram of Book Prices")
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.6)

ax = axes[1]
ax.boxplot(df["price_gbp"], vert=True, patch_artist=True,
           boxprops=dict(facecolor="#eef2ff", color=ACCENT),
           whiskerprops=dict(color=ACCENT), capprops=dict(color=ACCENT),
           medianprops=dict(color="#dc2626", linewidth=2),
           flierprops=dict(marker="o", markerfacecolor="#7c3aed", markersize=4, alpha=0.5))
ax.set_ylabel("Price (£)")
ax.set_title("Box Plot — Price Spread & Outliers")
ax.set_xticks([])
ax.grid(axis="y", alpha=0.6)

plt.tight_layout()
plt.savefig("chart1_price_distribution.png", bbox_inches="tight", dpi=130)
plt.close()
log("\n  [Saved] chart1_price_distribution.png")

# ── Chart 2: Rating Distribution ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(BG)
rc = df["rating"].value_counts().sort_index()
colors = ["#dc2626","#d97706","#eab308","#059669","#4f46e5"]
bars = ax.bar(rc.index, rc.values, color=colors, edgecolor=BG, linewidth=0.6, width=0.6)
for bar, val in zip(bars, rc.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f"{val}\n({100*val/len(df):.1f}%)", ha="center", fontsize=9, color=TEXT)
ax.set_xlabel("Star Rating")
ax.set_ylabel("Number of Books")
ax.set_title("Rating Distribution — How Books Are Rated", fontsize=13, fontweight="bold", color=TEXT)
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["1 ★","2 ★","3 ★","4 ★","5 ★"])
ax.set_ylim(0, rc.max() * 1.18)
ax.grid(axis="y", alpha=0.6)
plt.tight_layout()
plt.savefig("chart2_rating_distribution.png", bbox_inches="tight", dpi=130)
plt.close()
log("  [Saved] chart2_rating_distribution.png")

# ── Chart 3: Top 10 Categories by Book Count ──────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG)
top10 = df["category"].value_counts().head(10)
bars = ax.barh(top10.index[::-1], top10.values[::-1],
               color=[PALETTE[i % len(PALETTE)] for i in range(10)],
               edgecolor=BG, linewidth=0.4, height=0.65)
for bar, val in zip(bars, top10.values[::-1]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=9, color=TEXT)
ax.set_xlabel("Number of Books")
ax.set_title("Top 10 Categories by Book Count", fontsize=13, fontweight="bold", color=TEXT)
ax.set_xlim(0, top10.max() * 1.12)
ax.grid(axis="x", alpha=0.6)
plt.tight_layout()
plt.savefig("chart3_category_counts.png", bbox_inches="tight", dpi=130)
plt.close()
log("  [Saved] chart3_category_counts.png")

# ── Chart 4: Average Price per Category (top 15) ──────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG)
avg_price = df.groupby("category")["price_gbp"].mean().sort_values(ascending=False).head(15)
bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(avg_price))]
bars = ax.bar(range(len(avg_price)), avg_price.values,
              color=bar_colors, edgecolor=BG, linewidth=0.4, width=0.65)
for bar, val in zip(bars, avg_price.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"£{val:.1f}", ha="center", fontsize=8, color=TEXT)
ax.set_xticks(range(len(avg_price)))
ax.set_xticklabels(avg_price.index, rotation=38, ha="right", fontsize=9)
ax.set_ylabel("Average Price (£)")
ax.set_title("Average Book Price by Category (Top 15)", fontsize=13, fontweight="bold", color=TEXT)
ax.axhline(df["price_gbp"].mean(), color="#dc2626", linewidth=1.4, linestyle="--",
           label=f"Overall mean £{df['price_gbp'].mean():.2f}")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.6)
plt.tight_layout()
plt.savefig("chart4_avg_price_by_category.png", bbox_inches="tight", dpi=130)
plt.close()
log("  [Saved] chart4_avg_price_by_category.png")

# ── Chart 5: Correlation Heatmap ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(BG)
num_cols = ["price_gbp","rating","pages","year_published","num_reviews"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
cmap = sns.diverging_palette(20, 230, as_cmap=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
            linewidths=0.5, linecolor=BG,
            annot_kws={"size": 10, "color": TEXT},
            ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix — Numeric Features", fontsize=13, fontweight="bold", color=TEXT)
ax.tick_params(axis="both", labelsize=9)
plt.tight_layout()
plt.savefig("chart5_correlation_heatmap.png", bbox_inches="tight", dpi=130)
plt.close()
log("  [Saved] chart5_correlation_heatmap.png")

# ── Chart 6: Price vs Rating Scatter + Stock Status ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Bivariate Analysis — Price, Rating & Stock Status",
             fontsize=14, fontweight="bold", color=TEXT, y=1.01)

ax = axes[0]
for rating_val, grp in df.groupby("rating"):
    ax.scatter(grp["price_gbp"], grp["rating"] + np.random.uniform(-0.15, 0.15, len(grp)),
               alpha=0.35, s=18, color=colors[rating_val - 1], label=f"{rating_val}★")
m, b = np.polyfit(df["price_gbp"], df["rating"], 1)
x_line = np.linspace(df["price_gbp"].min(), df["price_gbp"].max(), 200)
ax.plot(x_line, m * x_line + b, color=TEXT, linewidth=1.4, linestyle="--", alpha=0.7, label="Trend")
ax.set_xlabel("Price (£)")
ax.set_ylabel("Rating (stars)")
ax.set_title("Price vs. Rating")
ax.set_yticks([1,2,3,4,5])
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.4)

ax = axes[1]
in_stock_avg  = df.groupby(["rating","in_stock"])["price_gbp"].mean().unstack()
x = np.arange(1, 6)
w = 0.38
bars_in  = ax.bar(x - w/2, in_stock_avg.get(True,  pd.Series([0]*5, index=range(1,6))),
                   width=w, color="#059669", label="In Stock",     edgecolor=BG)
bars_out = ax.bar(x + w/2, in_stock_avg.get(False, pd.Series([0]*5, index=range(1,6))),
                   width=w, color="#dc2626", label="Out of Stock", edgecolor=BG)
ax.set_xticks(x)
ax.set_xticklabels(["1★","2★","3★","4★","5★"])
ax.set_ylabel("Average Price (£)")
ax.set_title("Avg Price by Rating & Stock Status")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.6)

plt.tight_layout()
plt.savefig("chart6_price_rating_scatter.png", bbox_inches="tight", dpi=130)
plt.close()
log("  [Saved] chart6_price_rating_scatter.png")

# ── Write report file ─────────────────────────────────────────────────────────
with open("eda_report.txt", "w") as f:
    f.write("\n".join(report_lines))

log("\n" + "=" * 65)
log("  EDA Complete. Files saved:")
log("    eda_report.txt")
log("    chart1_price_distribution.png")
log("    chart2_rating_distribution.png")
log("    chart3_category_counts.png")
log("    chart4_avg_price_by_category.png")
log("    chart5_correlation_heatmap.png")
log("    chart6_price_rating_scatter.png")
log("=" * 65)
