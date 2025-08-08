import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# 1. 读取数据
train = pd.read_csv("./Data/input_data/df_movies_train.csv")
test = pd.read_csv("./Data/input_data/df_movies_test.csv")


# 2. 特征工程（结合你的EDA结论）
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
    # 制作公司、制片人数量
    df["production_companies_num"] = df["production_companies"].apply(
        lambda x: len(str(x).split(",")) if x not in [np.nan, None, "missing"] else 0
    )
    df["producers_num"] = df["producers"].apply(
        lambda x: len(str(x).split(",")) if x not in [np.nan, None, "missing"] else 0
    )
    # 导演、编剧数量
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

# 4. 类别特征编码（LightGBM支持整数编码，LabelEncoder最简单）
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

# 7. LightGBM模型训练
lgb_train = lgb.Dataset(
    X_train,
    label=y_train,
    categorical_feature=[all_features.index(col) for col in cat_cols],
)
lgb_val = lgb.Dataset(
    X_val,
    label=y_val,
    categorical_feature=[all_features.index(col) for col in cat_cols],
)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.08,
    "num_leaves": 31,
    "max_depth": 7,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "seed": 42,
    "verbose": -1,
}

# gbm = lgb.train(
#     params,
#     lgb_train,
#     num_boost_round=500,
#     valid_sets=[lgb_train, lgb_val],
#     valid_names=["train", "valid"],
#     early_stopping_rounds=30,
#     verbose_eval=50,
# )
# 当前LightGBM的版本（或你的 lgb.train 调用）不支持 early_stopping_rounds 作为关键字参数
try:
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        early_stopping_rounds=30,
        verbose_eval=50,
    )
except:
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=200,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
    )

# 8. 验证集RMSE
val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"验证集RMSE: {rmse:.4f}")

# 9. 测试集预测和结果保存
test_pred = gbm.predict(test[all_features], num_iteration=gbm.best_iteration)
result = pd.DataFrame({"id": test["id"], "rating": np.round(test_pred, 2)})
result.to_csv("./Data/output_result/df_result_1.csv", index=False)
print("已保存预测结果到 ./Data/output_result/df_result_1.csv")
