# 火花杯C题问题二：电影评分预测方法说明

## 方法概述

本项目针对火花杯C题问题二（评分预测），采用了业界主流的机器学习方法进行回归建模，并在模型性能上做了横向对比。

主要流程包括：

1. **数据预处理**
   - 缺失值处理：文本特征缺失用"missing"填充，数值型特征用均值填充。
   - 特征工程：结合数据探索结论，构建了如高评分类型、低评分类型、非英语语言、团队规模等衍生特征。

2. **特征编码**
   - 对类别型特征统一使用LabelEncoder编码，以适配各主流模型。

3. **模型训练与比较**
   - 采用了 CatBoost、XGBoost、LightGBM 三种集成树模型进行评分预测。
   - 使用 `train_test_split` 从训练数据中划分出20%作为验证集。
   - 以验证集RMSE（均方根误差）为主要模型评估指标，同时利用官方提供的测试集真实评分（df_movies_schedule.csv）对测试集预测性能进行评估。

4. **模型对比**
   - 输出三个模型在验证集和测试集上的RMSE，分别按训练和测试效果排序，方便横向分析和最佳模型选择。

---

## 使用代码的方法

1. **准备数据文件**
   - 将数据文件放置于如下目录（保持和代码一致）：
     ```
     ./Data/input_data/df_movies_train.csv
     ./Data/input_data/df_movies_test.csv
     ./Data/input_data/df_movies_schedule.csv
     ```
   
2. **运行Python代码**
   
    首先进入工作目录`Spark_Cup-2025-Question_C`
    
    - 要输出df_result_1.csv文件，运行以下命令，用**LightGBM模型**进行预测：
        ```bash
        python Question_2/Question_2.py
        ```

    - 要运行LightGBM模型，运行以下命令：
        ```bash
        python Question_2/LightGBM.py
        ```

    - 要运行CatBoost模型，运行以下命令：
        ```bash
        python Question_2/CatBoost.py
        ```

    - 要运行XGBoost模型，运行以下命令：
        ```bash
        python Question_2/XGBoost.py
        ```

    - 要比较不同模型的效果，运行以下命令：
        ```bash
        python Question_2/Model_Comparison.py
        ```

3. **主要输出解释**
   - **训练集RMSE**：模型在从训练集中分出的20%验证集上的误差，反映泛化能力。
   - **测试集RMSE**：模型在官方提供的真实测试集标签（df_movies_schedule.csv）上的误差，接近最终竞赛评分。
   - 排序输出便于选择效果最优的模型。

---


