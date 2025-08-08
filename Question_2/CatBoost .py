import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. 读取数据
train = pd.read_csv("./Data/input_data/df_movies_train.csv")
test = pd.read_csv("./Data/input_data/df_movies_test.csv")

# 2. 缺失值处理
cat_cols = [
    "genres",
    "cast",
    "director",
    "writers",
    "production_companies",
    "producers",
    "original_language",
]

num_cols = ["runtime"]

for col in cat_cols:
    train[col] = train[col].fillna("missing")
    test[col] = test[col].fillna("missing")

for col in num_cols:
    mean_val = train[col].mean()
    train[col] = train[col].fillna(mean_val)
    test[col] = test[col].fillna(mean_val)

# 3. 特征处理（可根据实际需要改进/丰富）
# 这里仅用runtime、original_language、genres、director等基础字段
# CatBoost支持字符串类别，直接用即可
feature_cols = ["runtime", "original_language", "genres", "director"]

# 4. 划分训练/验证集（仅用于本地调参）
X_train, X_val, y_train, y_val = train_test_split(
    train[feature_cols], train["rating"], test_size=0.2, random_state=42
)

# 5. CatBoost模型训练
cat_features = [feature_cols.index(col) for col in feature_cols if col in cat_cols]
model = CatBoostRegressor(
    iterations=200,
    learning_rate=0.1,
    depth=6,
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

# 6. 验证集评估
val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"验证集RMSE: {rmse:.4f}")

# 7. 对测试集预测
test_pred = model.predict(test[feature_cols])

# 8. 结果导出
result = pd.DataFrame(
    {"id": test["id"], "rating": np.round(test_pred, 2)}  # 评分保留两位小数
)
result.to_csv("./Data/output_result/df_result_1.csv", index=False)
print("已保存预测结果到 ./Data/output_result/df_result_1.csv")
