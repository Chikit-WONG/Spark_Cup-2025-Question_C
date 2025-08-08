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
test_truth = pd.read_csv("./Data/input_data/df_movies_schedule.csv")[["id", "rating"]]


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


def calc_test_rmse(model, test, all_features, test_truth, model_name):
    test_pred = model.predict(test[all_features])
    pred_df = pd.DataFrame({"id": test["id"], "pred": test_pred})
    merged = pd.merge(pred_df, test_truth, on="id", how="inner")
    rmse = np.sqrt(mean_squared_error(merged["rating"], merged["pred"]))
    return rmse


# 7. CatBoost
cat_features_idx = [all_features.index(col) for col in cat_cols if col in all_features]
cat_model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.08,
    depth=7,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
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
cat_test_rmse = calc_test_rmse(cat_model, test, all_features, test_truth, "CatBoost")
results["CatBoost"] = (cat_rmse, cat_test_rmse)

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
)
xgb_val_pred = xgb_model.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
xgb_test_rmse = calc_test_rmse(xgb_model, test, all_features, test_truth, "XGBoost")
results["XGBoost"] = (xgb_rmse, xgb_test_rmse)

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
)
lgbm.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    categorical_feature=cat_cols,
)
lgb_val_pred = lgbm.predict(X_val)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
lgb_test_rmse = calc_test_rmse(lgbm, test, all_features, test_truth, "LightGBM")
results["LightGBM"] = (lgb_rmse, lgb_test_rmse)

# ----------- 分别排序输出 ---------------

print("\n按训练集RMSE排序：")
for model, (train_rmse, test_rmse) in sorted(results.items(), key=lambda x: x[1][0]):
    print(f"{model:10s} 训练集RMSE: {train_rmse:.4f}   测试集RMSE: {test_rmse:.4f}")

print("\n按测试集RMSE排序：")
for model, (train_rmse, test_rmse) in sorted(results.items(), key=lambda x: x[1][1]):
    print(f"{model:10s} 测试集RMSE: {test_rmse:.4f}   训练集RMSE: {train_rmse:.4f}")
