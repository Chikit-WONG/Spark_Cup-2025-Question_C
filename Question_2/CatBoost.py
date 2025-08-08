import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. 读取数据
train = pd.read_csv("./Data/input_data/df_movies_train.csv")
test = pd.read_csv("./Data/input_data/df_movies_test.csv")

# 2. 统一所有输入特征
feature_cols = [col for col in train.columns if col not in ["id", "rating"]]

# 3. 缺失值处理
for col in feature_cols:
    if train[col].dtype == "object":
        train[col] = train[col].fillna("missing")
        test[col] = test[col].fillna("missing")
    else:
        mean_val = train[col].mean()
        train[col] = train[col].fillna(mean_val)
        test[col] = test[col].fillna(mean_val)

# 4. 按你的结论，提取/补充特征


# (1) 电影时长，直接保留原特征
# (2) 电影类型：音乐、纪录片、高分；恐怖片低分——> 增加“高评分类型”“低评分类型”二值特征
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


# (3) 语言分组：非英语高分
def is_non_english(lang):
    return 0 if lang == "en" or lang == "english" else 1


for df in [train, test]:
    df["non_english"] = df["original_language"].apply(is_non_english)

# (4) 制作公司数、制片人数
for df in [train, test]:
    # 统计分隔符个数+1（如果是字符串多公司/人以逗号或|分隔）
    for col in ["production_companies", "producers"]:
        df[f"{col}_num"] = df[col].apply(
            lambda x: (
                len(str(x).split(",")) if x not in [np.nan, None, "missing"] else 0
            )
        )

# (5) 导演/编剧人数
for df in [train, test]:
    for col in ["director", "writers"]:
        df[f"{col}_num"] = df[col].apply(
            lambda x: (
                len(str(x).split(",")) if x not in [np.nan, None, "missing"] else 0
            )
        )

# 5. 最终特征组合
# 原始数值+类别特征 + 你分析出的衍生特征
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

cat_cols = [
    "original_language",
    "genres",
    "director",
    "cast",
    "writers",
    "production_companies",
    "producers",
]

cat_features = [all_features.index(col) for col in cat_cols if col in all_features]

# 6. 划分训练/验证集
X_train, X_val, y_train, y_val = train_test_split(
    train[all_features], train["rating"], test_size=0.2, random_state=42
)

# 7. CatBoost训练
model = CatBoostRegressor(
    iterations=200,
    learning_rate=0.08,
    depth=7,
    loss_function="RMSE",
    verbose=50,
    random_seed=42,
)
model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_features,
    use_best_model=True,
)

# 8. 验证集RMSE
val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"验证集RMSE: {rmse:.4f}")

# 9. 测试集预测和结果保存
test_pred = model.predict(test[all_features])
result = pd.DataFrame({"id": test["id"], "rating": np.round(test_pred, 2)})
result.to_csv("./Data/output_result/df_result_1.csv", index=False)
print("已保存预测结果到 ./Data/output_result/df_result_1.csv")

# # 10. 计算测试集RMSE（根据真实标签）
# # 读取真实评分文件
# schedule = pd.read_csv("./Data/input_data/df_movies_schedule.csv")[["id", "rating"]]
# test_result = pd.DataFrame({"id": test["id"], "pred": test_pred})

# # 按 id 合并预测与真实值
# merged = pd.merge(test_result, schedule, on="id", how="inner")

# # 计算并输出测试集RMSE
# rmse_test = np.sqrt(mean_squared_error(merged["rating"], merged["pred"]))
# print(f"测试集RMSE: {rmse_test:.4f}")
# 看到比赛问答后发现，原来df_movies_schedule.csv中的rating是仅用于展示数据类型的随机值，无实际含义
