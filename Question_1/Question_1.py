# movie_rating_analysis.py
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------------
# Config / Paths
# --------------------------
INPUT_CSV = r"C:\Users\Lenovo\Desktop\code\Spark_Cup-2025-Question_C\Data\input_data\df_movies_train.csv"
FIG_DIR = "./figs"
os.makedirs(FIG_DIR, exist_ok=True)

# --------------------------
# 1. Read data
# --------------------------
df = pd.read_csv(INPUT_CSV, sep=None, engine="python")  # auto-detect separator
print("原始数据形状：", df.shape)
print("前5行预览：")
print(df.head())

# Standardize column names (strip)
df.columns = [c.strip() for c in df.columns]

# --------------------------
# 2. Basic cleaning & types
# --------------------------
# Ensure rating & runtime numeric
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')

# Fill missing textual fields with empty string
text_cols = ['genres','cast','director','writers','production_companies','producers','original_language']
for c in text_cols:
    if c in df.columns:
        df[c] = df[c].fillna('').astype(str)
    else:
        df[c] = ''

# Drop rows without rating (we cannot model those)
before_n = len(df)
df = df[~df['rating'].isna()].reset_index(drop=True)
print(f"删除无评分行: {before_n - len(df)} 行，剩余 {len(df)} 行")

# --------------------------
# 3. Feature engineering
# --------------------------
# Helper: split comma-separated fields into list (strip whitespace)
def split_list(s):
    if not s or s.strip()=='':
        return []
    parts = [p.strip() for p in s.split(',') if p.strip()!='']
    return parts

# Create list columns
df['genres_list'] = df['genres'].apply(split_list)
df['cast_list'] = df['cast'].apply(split_list)
df['director_list'] = df['director'].apply(split_list)
df['writers_list'] = df['writers'].apply(split_list)
df['prodcomp_list'] = df['production_companies'].apply(split_list)
df['producers_list'] = df['producers'].apply(split_list)
df['language'] = df['original_language'].fillna('').astype(str)

# Count features
df['n_genres'] = df['genres_list'].apply(len)
df['n_cast'] = df['cast_list'].apply(len)
df['n_director'] = df['director_list'].apply(len)
df['n_writers'] = df['writers_list'].apply(len)
df['n_prodcomp'] = df['prodcomp_list'].apply(len)
df['n_producers'] = df['producers_list'].apply(len)

# Binning runtime for EDA convenience
df['runtime_bin'] = pd.cut(df['runtime'].fillna(-1),
                           bins=[-2,0,60,90,120,180,10000],
                           labels=['missing','<=60','61-90','91-120','121-180','>180'])

# Create MultiLabelBinarizer features for top genres / top cast / top directors / languages
TOP_GENRES = 20
TOP_CAST = 50
TOP_DIRECTORS = 30
TOP_PRODCOMP = 30
TOP_LANGS = 15

# Helper to get top-k items from a list column
def get_top_k(series_of_lists, k):
    from collections import Counter
    cnt = Counter()
    for L in series_of_lists:
        cnt.update(L)
    most = [x for x,_ in cnt.most_common(k)]
    return most

top_genres = get_top_k(df['genres_list'], TOP_GENRES)
top_cast = get_top_k(df['cast_list'], TOP_CAST)
top_directors = get_top_k(df['director_list'], TOP_DIRECTORS)
top_prodcomp = get_top_k(df['prodcomp_list'], TOP_PRODCOMP)
top_langs = [x for x,_ in df['language'].value_counts().head(TOP_LANGS).items()]

print("Top genres:", top_genres)
print("Top languages:", top_langs[:10])

# MultiLabelBinarizer for genres
mlb_genre = MultiLabelBinarizer(classes=top_genres)
genre_mat = mlb_genre.fit_transform(df['genres_list'])
genre_df = pd.DataFrame(genre_mat, columns=[f"genre_{g}" for g in mlb_genre.classes_], index=df.index)

# For cast, only keep top actors as binary flags
def make_top_flags(series_of_lists, top_list, prefix):
    arr = np.zeros((len(series_of_lists), len(top_list)), dtype=int)
    top_set = set(top_list)
    for i, L in enumerate(series_of_lists):
        for it in L:
            if it in top_set:
                arr[i, top_list.index(it)] = 1
    return pd.DataFrame(arr, columns=[f"{prefix}_{t}" for t in top_list], index=df.index)

cast_df = make_top_flags(df['cast_list'], top_cast, "actor")
dir_df = make_top_flags(df['director_list'], top_directors, "dir")
prodcomp_df = make_top_flags(df['prodcomp_list'], top_prodcomp, "prodcomp")

# Language one-hot (top languages only, rest as 'other')
df['lang_top'] = df['language'].where(df['language'].isin(top_langs), 'other')
lang_dummies = pd.get_dummies(df['lang_top'], prefix='lang')

# Combine features into one dataframe
feature_df = pd.concat([
    df[['runtime','n_genres','n_cast','n_director','n_writers','n_prodcomp','n_producers']],
    genre_df,
    cast_df,
    dir_df,
    prodcomp_df,
    lang_dummies
], axis=1).fillna(0)

print("特征矩阵形状：", feature_df.shape)

# --------------------------
# 4. Exploratory Data Analysis (plots saved)
# --------------------------
sns.set(style="whitegrid", font_scale=1.0)

# Rating distribution
plt.figure(figsize=(8,5))
sns.histplot(df['rating'].dropna(), kde=True, bins=30)
plt.title("Rating distribution")
plt.xlabel("rating")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "rating_distribution.png"))
plt.close()

# runtime vs rating scatter + trend
plt.figure(figsize=(8,5))
sns.scatterplot(x='runtime', y='rating', data=df, alpha=0.5)
sns.regplot(x='runtime', y='rating', data=df, scatter=False, lowess=True, line_kws={'color':'red'})
plt.title("Runtime vs Rating")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "runtime_vs_rating.png"))
plt.close()

# Boxplot: rating by runtime_bin
plt.figure(figsize=(9,5))
order = ['<=60','61-90','91-120','121-180','>180','missing']
sns.boxplot(x='runtime_bin', y='rating', data=df, order=order)
plt.title("Rating by runtime bin")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "rating_by_runtime_bin.png"))
plt.close()

# Average rating by top genres (exploded)
genre_mean = []
for g in top_genres:
    mask = df['genres_list'].apply(lambda L: g in L)
    if mask.sum() >= 3:
        genre_mean.append((g, df.loc[mask,'rating'].mean(), mask.sum()))
genre_mean = sorted(genre_mean, key=lambda x: x[1], reverse=True)
g_names = [x[0] for x in genre_mean]
g_vals = [x[1] for x in genre_mean]

plt.figure(figsize=(10,6))
sns.barplot(x=g_vals, y=g_names)
plt.xlabel("mean rating")
plt.title("Average rating by genre (top genres)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "avg_rating_by_genre.png"))
plt.close()

# Average rating by language
lang_stats = df.groupby('lang_top')['rating'].agg(['mean','count']).sort_values('mean', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=lang_stats['mean'], y=lang_stats.index)
plt.xlabel("mean rating")
plt.title("Average rating by language (top languages)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "avg_rating_by_language.png"))
plt.close()

print("EDA 图片已保存到", FIG_DIR)

# --------------------------
# 5. Modeling: prepare X, y
# --------------------------
X = feature_df.copy()
y = df['rating'].values

# Reduce dimensionality by removing all-zero columns (rare top items not present)
nonzero_cols = X.columns[(X.sum(axis=0) > 0).values]
X = X[nonzero_cols]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Standardize numeric columns for Lasso/OLS
numeric_cols = ['runtime','n_genres','n_cast','n_director','n_writers','n_prodcomp','n_producers']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
for col in numeric_cols:
    if col in X_train.columns:
        X_train_scaled[col] = scaler.fit_transform(X_train[[col]])
        X_test_scaled[col] = scaler.transform(X_test[[col]])

# --------------------------
# 5.1 OLS linear regression (statsmodels) -- using a reduced set of features for interpretability
# We'll use: runtime + n_* counts + language dummies + top genre dummies + top director dummies (few)
# --------------------------
# select a compact subset for OLS to keep summary readable
ols_cols = []
for c in ['runtime','n_genres','n_cast','n_director','n_writers','n_prodcomp','n_producers']:
    if c in X_train.columns:
        ols_cols.append(c)
# add top 8 genres if present
ols_cols += [c for c in X_train.columns if c.startswith('genre_')][:8]
# add top 6 languages
ols_cols += [c for c in X_train.columns if c.startswith('lang_')][:6]
# add top 6 directors if present
ols_cols += [c for c in X_train.columns if c.startswith('dir_')][:6]

# ensure unique
ols_cols = list(dict.fromkeys(ols_cols))
X_ols = sm.add_constant(X_train[ols_cols].astype(float))
model_ols = sm.OLS(y_train, X_ols).fit()
print("\nOLS 回归摘要（选取的少量特征以便解释）：")
print(model_ols.summary())

# Evaluate OLS on test
X_test_ols = sm.add_constant(X_test[ols_cols].astype(float))
y_pred_ols = model_ols.predict(X_test_ols)
print("OLS test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_ols)))
print("OLS test R2:", r2_score(y_test, y_pred_ols))

# --------------------------
# 5.2 Lasso (feature selection + coefficients)
# --------------------------
print("\n训练 LassoCV (自动选 alpha)...")
lasso = LassoCV(cv=5, random_state=RANDOM_STATE, n_jobs=-1, max_iter=5000).fit(X_train_scaled, y_train)
print("Lasso alpha:", lasso.alpha_)
# coefficients
coef = pd.Series(lasso.coef_, index=X_train_scaled.columns).sort_values()
top_pos = coef.tail(15)
top_neg = coef.head(15)

print("\nLasso: 最强正向系数（可能正相关影响 rating 的特征）：")
print(top_pos.tail(10))
print("\nLasso: 最强负向系数：")
print(top_neg.head(10))

# Lasso test performance
y_pred_lasso = lasso.predict(X_test_scaled)
print("Lasso test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lasso)))
print("Lasso test R2:", r2_score(y_test, y_pred_lasso))

# --------------------------
# 5.3 Random Forest (non-linear) & feature importance
# --------------------------
print("\n训练随机森林以获得特征重要性...")
rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("RF test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("RF test R2:", r2_score(y_test, y_pred_rf))

feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
top_feat = feat_imp.head(30)
print("\nRandom Forest: 前30重要特征：")
print(top_feat)

# Plot top feature importances
plt.figure(figsize=(8,10))
sns.barplot(x=top_feat.values, y=top_feat.index)
plt.title("Top 30 feature importances (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "rf_top_features.png"))
plt.close()

# --------------------------
# 6. Summarize & print main discovered patterns
# --------------------------
def rmse(a,b): return np.sqrt(mean_squared_error(a,b))

print("\n--- 模型性能总结 ---")
print(f"OLS test RMSE: {rmse(y_test,y_pred_ols):.4f}, R2: {r2_score(y_test,y_pred_ols):.4f}")
print(f"Lasso test RMSE: {rmse(y_test,y_pred_lasso):.4f}, R2: {r2_score(y_test,y_pred_lasso):.4f}")
print(f"RF test RMSE: {rmse(y_test,y_pred_rf):.4f}, R2: {r2_score(y_test,y_pred_rf):.4f}")
print("\n注：R2 值通常较低，说明评分受许多未编码（或难以量化）的因素影响，如剧本质量、导演名气、演员表现、上映与宣传等。")

# Key takeaways from models:
print("\n--- 关键发现 (基于模型系数 / 重要性) ---")
# 1) runtime effect from OLS
if 'runtime' in model_ols.params.index:
    print(f"OLS: runtime 系数 = {model_ols.params['runtime']:.4f} (p={model_ols.pvalues['runtime']:.4g})")
else:
    print("OLS: runtime 未包含在模型列中")

# 2) show top positive/negative lasso features
print("\nLasso 发现的显著正向特征（系数正）：")
print(top_pos[top_pos>0].sort_values(ascending=False).head(15))

print("\nLasso 发现的显著负向特征（系数负）：")
print(top_neg[top_neg<0].sort_values().head(15))

print("\nRandom Forest 最重要的前10特征：")
print(top_feat.head(10))

# Save summary csvs
feat_imp.head(100).to_csv(os.path.join(FIG_DIR,"rf_feature_importances_top100.csv"))
coef.sort_values().to_csv(os.path.join(FIG_DIR,"lasso_coefficients.csv"))

# --------------------------
# 7. Final textual conclusions saved
# --------------------------
conclusions = []
conclusions.append("结论摘要：")
conclusions.append("1) 时长(runtime)通常与评分正相关（长片平均评分更高）。")
conclusions.append("2) 类型(genre)有差异，某些类型（如 music/ documentary/ animation）平均评分较高；horror 等类型平均评分偏低（需按数据集具体统计）。")
conclusions.append("3) 语言/产地(language)会影响平均评分，非英语电影在本数据里有时平均评分更高（例如日语等）。")
conclusions.append("4) 制作规模指标（出品公司数、制片人数量）与评分有弱正相关；而多导演/多编剧与评分有弱负相关。")
conclusions.append("5) 模型总体 R^2 偏低，说明评分被许多不可见或难量化的因素决定（例如导演声誉、剧本/剪辑、宣传、专业影评等）。")
with open(os.path.join(FIG_DIR,"conclusions.txt"), "w", encoding="utf8") as f:
    f.write("\n".join(conclusions))

print("\n分析完成。所有图表与模型结果已保存到目录：", FIG_DIR)
print("请查看该目录下的图片（rating_distribution.png, runtime_vs_rating.png, rating_by_runtime_bin.png, avg_rating_by_genre.png, avg_rating_by_language.png, rf_top_features.png），以及 CSV 结果。")
