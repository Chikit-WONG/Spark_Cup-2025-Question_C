# 模拟退火算法，小规模样本调通，只有核心约束
import pandas as pd
import numpy as np
import random
import copy

# ========== 1. 读入数据 ==========
df_cinema = pd.read_csv("./Data/input_data/df_cinema.csv")
df_movies = pd.read_csv("./Data/input_data/df_movies_schedule.csv")
df_rating = pd.read_csv("./Data/output_result/df_result_1.csv")
rating_dict = dict(zip(df_rating["id"], df_rating["rating"]))

rooms = df_cinema["room"].tolist()
movies = df_movies["id"].tolist()

# 电影信息预处理
movie_info = {}
for _, row in df_movies.iterrows():
    mid = row["id"]
    movie_info[mid] = {
        "runtime": int(np.ceil(row["runtime"] / 30) * 30),
        "basic_price": row["basic_price"],
        "versions": row["version"].split("|"),
        "original_language": row["original_language"],
    }

# 放映厅信息预处理
room_info = {}
for _, row in df_cinema.iterrows():
    rid = row["room"]
    room_info[rid] = {"capacity": row["capacity"], "versions": []}
    if row["2D"] == 1:
        room_info[rid]["versions"].append("2D")
    if row["3D"] == 1:
        room_info[rid]["versions"].append("3D")
    if row["IMAX"] == 1:
        room_info[rid]["versions"].append("IMAX")


# ========== 2. 工具函数 ==========
# 生成所有可用时间点
def generate_time_slots(start=600, end=1800, step=15):
    # 10:00=600，次日3:00=1800（单位分钟）
    return [i for i in range(start, end, step)]


time_slots = generate_time_slots()


def get_attendance(room, movie):
    return int(room_info[room]["capacity"] * rating_dict[movie] / 10)


def get_price(movie, version, t):
    base = movie_info[movie]["basic_price"]
    if version == "2D":
        price = base
    elif version == "3D":
        price = base * 1.2
    elif version == "IMAX":
        price = base * 1.23
    # 黄金时段票价上涨
    if 1080 <= t < 1260:
        price *= 1.3
    return price


def get_sharing_rate(movie):
    if "Mandarin" in movie_info[movie]["original_language"]:
        return 0.43
    else:
        return 0.51


def get_cost(room, version):
    cap = room_info[room]["capacity"]
    if version == "2D":
        coeff = 1.0
    elif version == "3D":
        coeff = 1.1
    elif version == "IMAX":
        coeff = 1.15
    return coeff * cap * 2.42 + 90


def format_time(minutes):
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


# 检查排片冲突
def is_feasible(schedule):
    # 保证同厅同时间不能有多场，且相邻电影间隔≥15min
    for room in schedule:
        times = []
        for show in schedule[room]:
            t, m, v = show["time"], show["movie"], show["version"]
            runtime = movie_info[m]["runtime"]
            times.append((t, t + runtime))
        times.sort()
        for i in range(1, len(times)):
            # 前一场结束后≥15min才能下一场
            if times[i][0] < times[i - 1][1] + 15:
                return False
    return True


# 计算净收益
def compute_objective(schedule):
    obj = 0
    for room in schedule:
        for show in schedule[room]:
            t, m, v = show["time"], show["movie"], show["version"]
            attendance = get_attendance(room, m)
            price = get_price(m, v, t)
            share = get_sharing_rate(m)
            cost = get_cost(room, v)
            obj += attendance * price * (1 - share) - cost
    return obj


# ========== 3. 初始可行解（贪心法，每厅每隔2小时随便排一部电影） ==========
def generate_initial_solution():
    schedule = {room: [] for room in rooms}
    for room in rooms:
        t = 600
        used_movies = set()
        while t + 60 <= 1800:  # 仅简单排片
            # 随机选支持版本的电影与版本
            candidates = []
            for m in movies:
                for v in movie_info[m]["versions"]:
                    if v in room_info[room]["versions"] and m not in used_movies:
                        candidates.append((m, v))
            if not candidates:
                break
            m, v = random.choice(candidates)
            schedule[room].append({"time": t, "movie": m, "version": v})
            used_movies.add(m)
            t += movie_info[m]["runtime"] + 15  # 播完休息15分钟
    return schedule


# ========== 4. 邻域操作（换一个厅的某场电影/时间/版本） ==========
def neighbor(schedule):
    # 深复制
    new_schedule = copy.deepcopy(schedule)
    # 随机选一个room和一个场次，换一部支持的电影
    room = random.choice(rooms)
    if not new_schedule[room]:
        return new_schedule
    idx = random.randint(0, len(new_schedule[room]) - 1)
    old = new_schedule[room][idx]
    # 换随机电影/版本/或微调时间
    actions = ["movie", "version", "time"]
    action = random.choice(actions)
    if action == "movie":
        candidates = []
        for m in movies:
            for v in movie_info[m]["versions"]:
                if v in room_info[room]["versions"]:
                    candidates.append((m, v))
        m, v = random.choice(candidates)
        new_schedule[room][idx]["movie"] = m
        new_schedule[room][idx]["version"] = v
    elif action == "version":
        m = old["movie"]
        versions = [
            v for v in movie_info[m]["versions"] if v in room_info[room]["versions"]
        ]
        new_schedule[room][idx]["version"] = random.choice(versions)
    elif action == "time":
        new_schedule[room][idx]["time"] = random.choice(time_slots)
    return new_schedule


# ========== 5. 模拟退火主流程 ==========
def simulated_annealing(max_iter=1000, T0=1000, Tmin=1e-3, alpha=0.95):
    curr = generate_initial_solution()
    while not is_feasible(curr):
        curr = generate_initial_solution()
    curr_score = compute_objective(curr)
    best, best_score = curr, curr_score
    T = T0
    for step in range(max_iter):
        for _ in range(20):  # 尝试20次邻域操作
            next_sol = neighbor(curr)
            if not is_feasible(next_sol):
                continue
            next_score = compute_objective(next_sol)
            dE = next_score - curr_score
            if dE > 0 or np.exp(dE / T) > random.random():
                curr, curr_score = next_sol, next_score
                if curr_score > best_score:
                    best, best_score = copy.deepcopy(curr), curr_score
        T *= alpha
        if T < Tmin:
            break
        if step % 50 == 0:
            print(f"Step {step}, Temp={T:.2f}, Best Net Profit={best_score:.2f}")
    return best, best_score


# ========== 6. 运行并输出 ==========
if __name__ == "__main__":
    best_schedule, best_score = simulated_annealing(
        max_iter=200, T0=500, Tmin=1e-2, alpha=0.96
    )
    print("最优净收益：", best_score)
    # 整理输出结果
    result = []
    for room in best_schedule:
        for show in best_schedule[room]:
            t, m, v = show["time"], show["movie"], show["version"]
            attendance = get_attendance(room, m)
            showtime = format_time(t)
            result.append([room, showtime, m, v, attendance])
    result_df = pd.DataFrame(
        result, columns=["room", "showtime", "id", "version", "attendance"]
    )
    result_df.sort_values(["room", "showtime"], inplace=True)
    result_df.to_csv("./Data/output_result/df_result_2.csv", index=False)
    print("已保存预测结果到 ./Data/output_result/df_result_2.csv")
    print(result_df)
