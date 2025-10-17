import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# LOAD DATA
# ============================================================================
data = pd.read_csv("/media/shomer/Windows/Users/shomer/data/New folder/genre_project/data/wiki_movie_plots_deduped.csv")

print("="*80)
print("MOVIE DATASET - EXPLORATORY DATA ANALYSIS")
print("="*80)

# ============================================================================
# BASIC INFO
# ============================================================================
print("\nDataset Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

print("\nColumn Types:")
print(data.dtypes)

print("\nMissing Values:")
print(data.isnull().sum())

print("\nBasic Statistics:")
print(data.describe())

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_bar(data, column, title, top_n=15, figsize=(12, 6)):
    """Simple bar plot for categorical data"""
    plt.figure(figsize=figsize)

    counts = data[column].value_counts().head(top_n)
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # ensure plots dir exists
    plots_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'plots')
    plots_dir = os.path.abspath(plots_dir)
    os.makedirs(plots_dir, exist_ok=True)

    # sanitize title for filename
    fn = re.sub(r"[^0-9a-zA-Z-_]+", '_', title).strip('_') + '.png'
    save_path = os.path.join(plots_dir, fn)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved plot: {save_path}")
    plt.show()

def plot_horizontal_bar(data, column, title, top_n=20, figsize=(10, 8)):
    """Horizontal bar plot for long labels"""
    plt.figure(figsize=figsize)

    counts = data[column].value_counts().head(top_n)
    sns.barplot(y=counts.index, x=counts.values, palette='rocket')

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.tight_layout()

    # ensure plots dir exists
    plots_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'plots')
    plots_dir = os.path.abspath(plots_dir)
    os.makedirs(plots_dir, exist_ok=True)

    fn = re.sub(r"[^0-9a-zA-Z-_]+", '_', title).strip('_') + '.png'
    save_path = os.path.join(plots_dir, fn)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved plot: {save_path}")
    plt.show()

def plot_line(data, column, title, figsize=(12, 6)):
    """Line plot for time series"""
    plt.figure(figsize=figsize)

    counts = data[column].value_counts().sort_index()
    plt.plot(counts.index, counts.values, linewidth=2, color='steelblue')

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Number of Movies', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # ensure plots dir exists
    plots_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'plots')
    plots_dir = os.path.abspath(plots_dir)
    os.makedirs(plots_dir, exist_ok=True)

    fn = re.sub(r"[^0-9a-zA-Z-_]+", '_', title).strip('_') + '.png'
    save_path = os.path.join(plots_dir, fn)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved plot: {save_path}")
    plt.show()


# ============================================================================
# PLOTS
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# 1. Movies by Origin/Ethnicity
print("\n1. Movies by Origin/Ethnicity")
plot_bar(data, 'Origin/Ethnicity', 'Movies by Origin/Ethnicity', top_n=15)

# 2. Movies over time
print("\n2. Movies Released Over Time")
plot_line(data, 'Release Year', 'Movies Released by Year')

# 3. Top movie titles
print("\n4. Most Common Movie Titles")
plot_bar(data, 'Title', 'Top 15 Movie Titles', top_n=15)

# 4. Most frequent genres
print("\n5. Most Frequent Genres")
data_genre = data[data['Genre'] != 'unknown']
plot_bar(data_genre, 'Genre', 'Top 15 Movie Genres', top_n=15)

# 5. Top cast members
print("\n6. Top Cast Members")
data_cast = data[(data['Cast'].notna()) & (data['Cast'] != ' ')]
plot_horizontal_bar(data_cast, 'Cast', 'Top 20 Cast Members', top_n=20)

# 6. Top directors
print("\n7. Top Directors")
data_dir = data[data['Director'] != 'Unknown']
plot_bar(data_dir, 'Director', 'Top 20 Directors', top_n=20, figsize=(14, 6))

print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)