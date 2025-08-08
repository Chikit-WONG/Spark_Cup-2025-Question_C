import pandas as pd
import pulp
import numpy as np

# 路径
cinema_path = "./Data/input_data/df_cinema.csv"
movies_path = "./Data/input_data/df_movies_schedule.csv"
output_path = "./Data/output_result/df_result_2.csv"

# 读取数据
df_cinema = pd.read_csv(cinema_path)
df_movies = pd.read_csv(movies_path)

# 假设已经有每部电影的预测评分，rating字段
# 以下生成所有可能的放映场次组合（示意，具体需根据你的数据规模精简处理）
rooms = df_cinema["room"].tolist()
movies = df_movies["id"].tolist()
versions = ["2D", "3D", "IMAX"]  # 仅做示例，具体按电影支持的版本和厅支持的版本生成
# 生成所有可选时间点（10:00-次日3:00，15min一档，共68档），以分钟计
time_slots = [600 + 15 * i for i in range(0, 68)]  # 10:00为第600分钟

# 预处理每部电影的可排版本、时长(30min整)、票价、类型等
movie_info = {}
for _, row in df_movies.iterrows():
    id = row["id"]
    movie_info[id] = {
        "runtime": int(np.ceil(row["runtime"] / 30) * 30),
        "rating": row["rating"],
        "basic_price": row["basic_price"],
        "genres": row["genres"],
        "original_language": row["original_language"],
        "versions": row["version"].split("|"),  # 假设是字符串用'|'分割
    }

# 预处理每个放映厅的可播版本
room_info = {}
for _, row in df_cinema.iterrows():
    room = row["room"]
    room_info[room] = {"capacity": row["capacity"], "versions": []}
    if row["2D"] == 1:
        room_info[room]["versions"].append("2D")
    if row["3D"] == 1:
        room_info[room]["versions"].append("3D")
    if row["IMAX"] == 1:
        room_info[room]["versions"].append("IMAX")

# 定义模型
prob = pulp.LpProblem("Cinema_Scheduling", pulp.LpMaximize)

# 决策变量（场次是否安排），变量量较大，数据规模太大时要裁剪！
x = {}
for r in rooms:
    for t in time_slots:
        for m in movies:
            for v in movie_info[m]["versions"]:
                if v in room_info[r]["versions"]:
                    x[r, t, m, v] = pulp.LpVariable(f"x_{r}_{t}_{m}_{v}", cat="Binary")


# 计算参数辅助函数
def get_attendance(r, m):
    return int(room_info[r]["capacity"] * movie_info[m]["rating"] / 10)


def get_price(m, v, t):
    base = movie_info[m]["basic_price"]
    if v == "2D":
        price = base
    elif v == "3D":
        price = base * 1.2
    elif v == "IMAX":
        price = base * 1.23
    # 黄金时段18:00-21:00，按分钟判断
    if 1080 <= t < 1260:  # 18:00~21:00
        price *= 1.3
    return price


def get_sharing_rate(m):
    # 原始语言含Mandarin为国产
    if "Mandarin" in movie_info[m]["original_language"]:
        return 0.43
    else:
        return 0.51


def get_cost(r, v):
    cap = room_info[r]["capacity"]
    if v == "2D":
        coeff = 1.0
    elif v == "3D":
        coeff = 1.1
    elif v == "IMAX":
        coeff = 1.15
    basic_cost = 2.42
    fixed_cost = 90
    return coeff * cap * basic_cost + fixed_cost


# 目标函数：最大化净收益
prob += pulp.lpSum(
    x[r, t, m, v]
    * (
        get_attendance(r, m) * get_price(m, v, t) * (1 - get_sharing_rate(m))
        - get_cost(r, v)
    )
    for (r, t, m, v) in x
)

# 约束1：每个厅每个时间点最多放一场电影
for r in rooms:
    for t in time_slots:
        prob += (
            pulp.lpSum(
                x[r, t, m, v]
                for m in movies
                for v in movie_info[m]["versions"]
                if v in room_info[r]["versions"]
                if (r, t, m, v) in x
            )
            <= 1
        )

# 约束2：同一厅排片间隔
for r in rooms:
    for t in time_slots:
        for m in movies:
            for v in movie_info[m]["versions"]:
                if (r, t, m, v) in x:
                    end_time = t + movie_info[m]["runtime"]
                    # 保证之后15分钟时间窗内没有其他排片
                    for t2 in time_slots:
                        if t2 > t and t2 < end_time + 15:
                            for m2 in movies:
                                for v2 in movie_info[m2]["versions"]:
                                    if (r, t2, m2, v2) in x:
                                        prob += x[r, t, m, v] + x[r, t2, m2, v2] <= 1

# （其他约束：如版本、时长上限、类型上下限、时段限制等请类似添加）

# 求解
prob.solve()

# 输出
result = []
for r, t, m, v in x:
    if x[(r, t, m, v)].varValue == 1:
        attendance = get_attendance(r, m)
        hour = int(t // 60)
        minute = int(t % 60)
        showtime = f"{hour:02d}:{minute:02d}"
        result.append([r, showtime, m, v, attendance])
result_df = pd.DataFrame(
    result, columns=["room", "showtime", "id", "version", "attendance"]
)
result_df.to_csv(output_path, index=False)
print(f"已保存预测结果到 {output_path}")
