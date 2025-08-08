import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# 1. 读取数据
train = pd.read_csv("./Data/input_data/df_movies_train.csv")
test = pd.read_csv("./Data/input_data/df_movies_test.csv")


# 2. 特征工程，结合你的EDA结论
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

# 4. 类别特征编码（LabelEncoder，每一列分开编码）
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

# 5. 选择最终特征
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

# 7. XGBoost模型训练
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="auto",
)
# xgb_model.fit(
#     X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=50
# )
#  XGBoost 的某些版本（特别是较老的 scikit-learn 风格 API 版本）不支持 early_stopping_rounds 作为 fit 方法的参数
try:
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=30,
        verbose=50,
    )
except TypeError:
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)


# 8. 验证集RMSE输出
val_pred = xgb_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"验证集RMSE: {rmse:.4f}")

# 9. 测试集预测和结果保存
test_pred = xgb_model.predict(test[all_features])
result = pd.DataFrame({"id": test["id"], "rating": np.round(test_pred, 2)})
result.to_csv("./Data/output_result/df_result_1.csv", index=False)
print("已保存预测结果到 ./Data/output_result/df_result_1.csv")

# # 10. 计算测试集RMSE（用df_movies_schedule.csv的真实标签）
# schedule = pd.read_csv("./Data/input_data/df_movies_schedule.csv")[["id", "rating"]]
# test_result = pd.DataFrame({"id": test["id"], "pred": test_pred})
# merged = pd.merge(test_result, schedule, on="id", how="inner")
# rmse_test = np.sqrt(mean_squared_error(merged["rating"], merged["pred"]))
# print(f"测试集RMSE: {rmse_test:.4f}")
# 看到比赛问答后发现，原来df_movies_schedule.csv中的rating是仅用于展示数据类型的随机值，无实际含义
