import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# config
# ======================
CSV_PATH = "./data/vggsound.csv"     # ← 改成你的路径
OUT_DIR = Path("./drafts/vggsound_stats")
OUT_DIR.mkdir(exist_ok=True)

TOPK = 309   # 只画前K个类（否则309类太密）


# ======================
# load
# ======================
df = pd.read_csv(CSV_PATH, header=None, names=["youtube_id", "start", "label", "split"])

print("Total samples:", len(df))
print("Splits:", df["split"].value_counts().to_dict())
print("Total classes:", df["label"].nunique())


# ======================
# helper function
# ======================
def plot_distribution(counts, title, filename):
    counts = counts.sort_values(ascending=False)

    # 保存完整csv
    counts.to_csv(OUT_DIR / f"{filename}_counts.csv")

    # 只画topK
    top_counts = counts.head(TOPK)

    plt.figure(figsize=(12, 6))
    top_counts.plot(kind="bar")
    plt.title(title + f" (Top {TOPK})")
    plt.ylabel("num samples")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{filename}.png", dpi=200)
    plt.close()


# ======================
# overall distribution
# ======================
overall_counts = df["label"].value_counts()
plot_distribution(overall_counts,
                  "VGGSound Overall Class Distribution",
                  "overall_distribution")


# ======================
# train distribution
# ======================
train_df = df[df["split"] == "train"]
train_counts = train_df["label"].value_counts()

plot_distribution(train_counts,
                  "Train Distribution",
                  "train_distribution")


# ======================
# test distribution
# ======================
test_df = df[df["split"] == "test"]
test_counts = test_df["label"].value_counts()

plot_distribution(test_counts,
                  "Test Distribution",
                  "test_distribution")


# ======================
# statistics summary
# ======================
stats = pd.DataFrame({
    "overall": overall_counts,
    "train": train_counts,
    "test": test_counts
}).fillna(0).astype(int)

stats.to_csv(OUT_DIR / "full_statistics.csv")

print("\nSaved to folder:", OUT_DIR)
print("Files generated:")
print("  overall_distribution.png")
print("  train_distribution.png")
print("  test_distribution.png")
print("  full_statistics.csv")
