import pandas as pd
import pulp
import numpy as np

# ====== 1. 小样本数据（可直接复制运行） ======

# 放映厅
df_cinema = pd.DataFrame(
    [
        {"room": "R1", "capacity": 100, "2D": 1, "3D": 1, "IMAX": 0},
        {"room": "R2", "capacity": 80, "2D": 1, "3D": 0, "IMAX": 0},
    ]
)

# 电影
df_movies = pd.DataFrame(
    [
        {
            "id": 1,
            "genres": "Action",
            "original_language": "Mandarin",
            "version": "2D|3D",
            "runtime": 100,
            "rating": 8.0,
            "basic_price": 50,
        },
        {
            "id": 2,
            "genres": "Drama",
            "original_language": "English",
            "version": "2D",
            "runtime": 90,
            "rating": 7.2,
            "basic_price": 45,
        },
        {
            "id": 3,
            "genres": "Animation",
            "original_language": "Mandarin",
            "version": "2D",
            "runtime": 80,
            "rating": 7.5,
            "basic_price": 40,
        },
    ]
)

# 时间段（10:00、12:00、14:00、16:00、18:00、20:00），按分钟
time_slots = [600, 720, 840, 960, 1080, 1200]

rooms = df_cinema["room"].tolist()
movies = df_movies["id"].tolist()

# ====== 2. 预处理 ======
# 电影信息
movie_info = {}
for _, row in df_movies.iterrows():
    m = row["id"]
    movie_info[m] = {
        "runtime": int(np.ceil(row["runtime"] / 30) * 30),
        "rating": row["rating"],
        "basic_price": row["basic_price"],
        "genres": row["genres"],
        "original_language": row["original_language"],
        "versions": row["version"].split("|"),
    }
# 放映厅信息
room_info = {}
for _, row in df_cinema.iterrows():
    r = row["room"]
    room_info[r] = {"capacity": row["capacity"], "versions": []}
    if row["2D"] == 1:
        room_info[r]["versions"].append("2D")
    if row["3D"] == 1:
        room_info[r]["versions"].append("3D")
    if row["IMAX"] == 1:
        room_info[r]["versions"].append("IMAX")

# ====== 3. 定义模型 ======
prob = pulp.LpProblem("Mini_Cinema_Scheduling", pulp.LpMaximize)

# 决策变量
x = {}
for r in rooms:
    for t in time_slots:
        for m in movies:
            for v in movie_info[m]["versions"]:
                if v in room_info[r]["versions"]:
                    x[r, t, m, v] = pulp.LpVariable(f"x_{r}_{t}_{m}_{v}", cat="Binary")


# ====== 4. 参数函数 ======
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
    # 黄金时段18:00-21:00
    if 1080 <= t < 1260:
        price *= 1.3
    return price


def get_sharing_rate(m):
    return 0.43 if "Mandarin" in movie_info[m]["original_language"] else 0.51


def get_cost(r, v):
    cap = room_info[r]["capacity"]
    if v == "2D":
        coeff = 1.0
    elif v == "3D":
        coeff = 1.1
    elif v == "IMAX":
        coeff = 1.15
    return coeff * cap * 2.42 + 90


# ====== 5. 目标函数 ======
prob += pulp.lpSum(
    x[r, t, m, v]
    * (
        get_attendance(r, m) * get_price(m, v, t) * (1 - get_sharing_rate(m))
        - get_cost(r, v)
    )
    for (r, t, m, v) in x
)

# ====== 6. 主要约束 ======
# 每厅每时刻最多一场
for r in rooms:
    for t in time_slots:
        prob += (
            pulp.lpSum(
                x[r, t, m, v]
                for m in movies
                for v in movie_info[m]["versions"]
                if v in room_info[r]["versions"] and (r, t, m, v) in x
            )
            <= 1
        )

# 同厅场次间无重叠（保证间隔，简化版，不考虑清场，仅保证不重叠）
for r in rooms:
    for t in time_slots:
        for m in movies:
            for v in movie_info[m]["versions"]:
                if (r, t, m, v) in x:
                    dur = movie_info[m]["runtime"]
                    end_t = t + dur
                    for t2 in time_slots:
                        if t2 > t and t2 < end_t:
                            for m2 in movies:
                                for v2 in movie_info[m2]["versions"]:
                                    if (r, t2, m2, v2) in x:
                                        prob += x[r, t, m, v] + x[r, t2, m2, v2] <= 1

# ====== 7. 求解 ======
prob.solve()

# ====== 8. 输出 ======
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
print(result_df)
result_df.to_csv("./Question_3/排片小样本结果.csv", index=False)
print("已保存预测结果到 ./Question_3/排片小样本结果.csv")
