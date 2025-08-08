import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 1. 读取数据
train = pd.read_csv("./Data/input_data/df_movies_train.csv")
test = pd.read_csv("./Data/input_data/df_movies_test.csv")


# 2. 特征工程（和前文一致，按你的结论来）
def type_group(genres, group):
    if pd.isna(genres):
        return 0
    genres = genres.lower()
    if any(g in genres for g in group):
        return 1
    return 0


high_score_types = ["music", "documentary"]
low_score_types = ["horror"]

for df in [train, test]:
    df["high_score_type"] = df["genres"].apply(
        lambda x: type_group(x, high_score_types)
    )
    df["low_score_type"] = df["genres"].apply(lambda x: type_group(x, low_score_types))
    df["non_english"] = df["original_language"].apply(
        lambda x: 0 if x == "en" or x == "english" else 1
    )
    df["production_companies_num"] = df["production_companies"].apply(
        lambda x: len(str(x).split(",")) if x not in [np.nan, None, "missing"] else 0
    )
    df["producers_num"] = df["producers"].apply(
        lambda x: len(str(x).split(",")) if x not in [np.nan, None, "missing"] else 0
    )
    df["director_num"] = df["director"].apply(
        lambda x: len(str(x).split(",")) if x not in [np.nan, None, "missing"] else 0
    )
    df["writers_num"] = df["writers"].apply(
        lambda x: len(str(x).split(",")) if x not in [np.nan, None, "missing"] else 0
    )

# 3. 缺失值处理
feature_cols = [col for col in train.columns if col not in ["id", "rating"]]
for col in feature_cols:
    if train[col].dtype == "object":
        train[col] = train[col].fillna("missing")
        test[col] = test[col].fillna("missing")
    else:
        mean_val = train[col].mean()
        train[col] = train[col].fillna(mean_val)
        test[col] = test[col].fillna(mean_val)

# 4. 类别特征处理
cat_cols = [
    "original_language",
    "genres",
    "director",
    "cast",
    "writers",
    "production_companies",
    "producers",
]
for col in cat_cols:
    le = LabelEncoder()
    all_values = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(all_values)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# 5. 最终特征
all_features = [
    "runtime",
    "original_language",
    "genres",
    "director",
    "cast",
    "writers",
    "production_companies",
    "producers",
    "high_score_type",
    "low_score_type",
    "non_english",
    "production_companies_num",
    "producers_num",
    "director_num",
    "writers_num",
]

# 6. 划分训练/验证集
X_train, X_val, y_train, y_val = train_test_split(
    train[all_features], train["rating"], test_size=0.2, random_state=42
)

results = {}

# 7. CatBoost
cat_features_idx = [all_features.index(col) for col in cat_cols if col in all_features]
cat_model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.08,
    depth=7,
    loss_function="RMSE",
    # verbose=0,
    random_seed=42,
)
cat_model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_features_idx,
    use_best_model=True,
)
cat_val_pred = cat_model.predict(X_val)
cat_rmse = np.sqrt(mean_squared_error(y_val, cat_val_pred))
results["CatBoost"] = cat_rmse

# 8. XGBoost
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="auto",
    verbosity=0,
)
xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    # verbose=0,
)
xgb_val_pred = xgb_model.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
results["XGBoost"] = xgb_rmse

# 9. LightGBM
lgbm = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.08,
    num_leaves=31,
    max_depth=7,
    feature_fraction=0.8,
    bagging_fraction=0.9,
    bagging_freq=1,
    random_state=42,
    # verbose=-1,
)
lgbm.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    # early_stopping_rounds=30,
    categorical_feature=cat_cols,
    # verbose=0,
)
lgb_val_pred = lgbm.predict(X_val)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
results["LightGBM"] = lgb_rmse

# 10. 输出横向对比结果
print("\n模型对比（RMSE越小越好）:")
for model, score in sorted(results.items(), key=lambda x: x[1]):
    print(f"{model:10s}: {score:.4f}")

# 你可以根据 RMSE 结果，选择最佳模型用于测试集预测与最终输出。
