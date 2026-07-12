import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f9f9f7",
    "axes.grid": True,
    "grid.color": "#e0dedd",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
 
PALETTE = {
    1: "#E24B4A",
    2: "#EF9F27",
    3: "#888780",
    4: "#1D9E75",
    5: "#185FA5",
}
COLOR_MAIN = "#534AB7"
COLOR_SEC  = "#1D9E75"

RESULTS_DIR = Path(__file__).resolve().parents[1] / "EDA results"
RESULTS_DIR.mkdir(exist_ok=True)

def _save_plot(filename: str):
    """Guarda la figura actual en la carpeta EDA results."""
    save_path = RESULTS_DIR / filename
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return save_path


def plot_score_distribution(df: pd.DataFrame):
    """Histograma / barras de Score (1-5 estrellas)."""
    counts = df["Score"].value_counts().sort_index()
    colors = [PALETTE[i] for i in counts.index]
 
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.6, edgecolor="white")
 
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{val:,}",
            ha="center", va="bottom", fontsize=9, color="#444"
        )
 
    pct = counts / counts.sum() * 100
    ax.set_title("Distribución de Score (estrellas)")
    ax.set_xlabel("Score")
    ax.set_ylabel("Cantidad de reseñas")
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f"★{i}\n{pct[i]:.1f}%" for i in range(1, 6)])
    plt.tight_layout()
    _save_plot("01_score_distribution.png")
    plt.show()

def plot_text_length_distribution(df: pd.DataFrame):
    """Histograma de text_length con líneas de percentiles."""
    fig, ax = plt.subplots(figsize=(8, 4))
 
    data = df["review_length"].clip(upper=df["review_length"].quantile(0.99))
    ax.hist(data, bins=60, color=COLOR_MAIN, alpha=0.8, edgecolor="white")
 
    for pct, label, ls in [
        (50, "Mediana", "--"),
        (90, "p90", ":"),
    ]:
        val = df["review_length"].quantile(pct / 100)
        ax.axvline(val, color="#E24B4A", linestyle=ls, linewidth=1.4,
                   label=f"{label}: {val:.0f} chars")
 
    ax.set_title("Distribución de longitud del texto (review_length)")
    ax.set_xlabel("Caracteres")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_plot("02_text_length_distribution.png")
    plt.show()

def plot_reviews_over_time(df: pd.DataFrame):
    """Serie temporal: cantidad de reseñas por mes."""
    monthly = df.groupby(pd.Grouper(key="Time", freq="ME")).size().reset_index(name="count")
 
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.fill_between(monthly["Time"], monthly["count"],
                    alpha=0.25, color=COLOR_MAIN)
    ax.plot(monthly["Time"], monthly["count"],
            color=COLOR_MAIN, linewidth=1.8)
 
    ax.set_title("Cantidad de reseñas a lo largo del tiempo")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Reseñas por mes")
    plt.tight_layout()
    _save_plot("03_reviews_over_time.png")
    plt.show()


def plot_top_products(df: pd.DataFrame, top_n: int = 20):
    """Top N productos con más reseñas."""
    top = df["ProductId"].value_counts().head(top_n)
 
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(top.index[::-1], top.values[::-1],
                   color=COLOR_SEC, edgecolor="white")
 
    for bar, val in zip(bars, top.values[::-1]):
        ax.text(val + top.max() * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)
 
    ax.set_title(f"Top {top_n} productos con más reseñas")
    ax.set_xlabel("Cantidad de reseñas")
    ax.set_ylabel("ProductId")
    plt.tight_layout()
    _save_plot("04_top_products.png")
    plt.show()
 
 
def plot_top_users(df: pd.DataFrame, top_n: int = 20):
    """Top N usuarios más activos."""
    top = df["UserId"].value_counts().head(top_n)
 
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index[::-1], top.values[::-1],
            color="#EF9F27", edgecolor="white")
 
    ax.set_title(f"Top {top_n} usuarios más activos")
    ax.set_xlabel("Cantidad de reseñas")
    ax.set_ylabel("UserId")
    plt.tight_layout()
    _save_plot("05_top_users.png")
    plt.show()
 
 
def plot_reviews_per_user_distribution(df: pd.DataFrame):
    """¿Cuántas reseñas escribe cada usuario? Distribución de frecuencias."""
    reviews_per_user = df["UserId"].value_counts()
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    axes[0].hist(reviews_per_user, bins=50,
                 color=COLOR_MAIN, alpha=0.8, edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_title("Reviews por usuario (escala log)")
    axes[0].set_xlabel("Número de reseñas")
    axes[0].set_ylabel("Cantidad de usuarios (log)")
 
    buckets = pd.cut(reviews_per_user,
                     bins=[0, 1, 5, 10, 50, reviews_per_user.max()],
                     labels=["1", "2-5", "6-10", "11-50", "50+"])
    bc = buckets.value_counts().sort_index()
    axes[1].bar(bc.index.astype(str), bc.values,
                color=COLOR_MAIN, alpha=0.8, edgecolor="white")
    axes[1].set_title("Usuarios agrupados por cantidad de reseñas")
    axes[1].set_xlabel("Reseñas escritas")
    axes[1].set_ylabel("Usuarios")
 
    plt.tight_layout()
    _save_plot("06_reviews_per_user.png")
    plt.show()

def plot_helpfulness_ratio_by_score(df: pd.DataFrame):
    """Boxplot del helpfulness_ratio por Score."""
    df_h = df[df["HelpfulnessDenominator"] > 0].copy()
 
    fig, ax = plt.subplots(figsize=(8, 5))
    data_by_score = [df_h[df_h["Score"] == s]["helpfulness_ratio"].dropna()
                     for s in range(1, 6)]
 
    bp = ax.boxplot(data_by_score, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
 
    for patch, score in zip(bp["boxes"], range(1, 6)):
        patch.set_facecolor(PALETTE[score])
        patch.set_alpha(0.8)
 
    ax.set_title("Helpfulness ratio por puntuación")
    ax.set_xlabel("Score (estrellas)")
    ax.set_ylabel("Helpfulness ratio (num / den)")
    ax.set_xticklabels([f"★{i}" for i in range(1, 6)])
    plt.tight_layout()
    _save_plot("07_helpfulness_by_score.png")
    plt.show()
 
 
def plot_denominator_distribution(df: pd.DataFrame):
    """Distribución de HelpfulnessDenominator (votos totales recibidos)."""
    denom = df["HelpfulnessDenominator"]
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    no_votes = (denom == 0).sum()
    has_votes = (denom > 0).sum()
    axes[0].pie([no_votes, has_votes],
                labels=[f"Sin votos\n{no_votes:,}", f"Con votos\n{has_votes:,}"],
                colors=["#D3D1C7", COLOR_SEC],
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 10})
    axes[0].set_title("Reseñas con y sin votos de helpfulness")
 
    d = denom[denom > 0].clip(upper=denom.quantile(0.99))
    axes[1].hist(d, bins=50, color=COLOR_SEC, alpha=0.8, edgecolor="white")
    axes[1].set_title("Distribución de votos recibidos (excl. 0)")
    axes[1].set_xlabel("Votos totales recibidos")
    axes[1].set_ylabel("Frecuencia")
 
    plt.tight_layout()
    _save_plot("08_denominator_distribution.png")
    plt.show()
 
 
def plot_text_length_vs_helpfulness(df: pd.DataFrame, sample_n: int = 5000):
    """Scatter: text_length vs. helpfulness_ratio coloreado por Score."""
    df_h = df[(df["HelpfulnessDenominator"] > 0) &
              df["helpfulness_ratio"].notna()].copy()
 
    df_sample = df_h.sample(min(sample_n, len(df_h)), random_state=42)
 
    fig, ax = plt.subplots(figsize=(8, 5))
    for score in range(1, 6):
        mask = df_sample["Score"] == score
        ax.scatter(
            df_sample.loc[mask, "review_length"].clip(upper=3000),
            df_sample.loc[mask, "helpfulness_ratio"],
            alpha=0.3, s=15, color=PALETTE[score], label=f"★{score}"
        )
 
    ax.set_title(f"Longitud del texto vs. helpfulness ratio (muestra {sample_n:,})")
    ax.set_xlabel("text_length (chars, cap 3000)")
    ax.set_ylabel("Helpfulness ratio")
    ax.legend(title="Score", fontsize=8, title_fontsize=8,
              loc="upper right", markerscale=2)
    plt.tight_layout()
    _save_plot("09_length_vs_helpfulness.png")
    plt.show()