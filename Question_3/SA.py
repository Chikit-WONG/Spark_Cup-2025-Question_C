# 模拟退火算法，全数据批量运行，所有条件约束
import pandas as pd
import numpy as np
import random
import copy
from tqdm import tqdm

# ========== 1. 读入数据 ==========
cinema_path = "./Data/input_data/df_cinema.csv"
movies_path = "./Data/input_data/df_movies_schedule.csv"
rating_path = "./Data/output_result/df_result_1.csv"
output_path = "./Data/output_result/df_result_2.csv"

df_cinema = pd.read_csv(cinema_path)
df_movies = pd.read_csv(movies_path)
df_rating = pd.read_csv(rating_path)
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
        "genres": row["genres"],
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


# ========== 2. 全部可用时间点 ==========
def generate_time_slots(start=600, end=1800, step=15):
    return [i for i in range(start, end, step)]


time_slots = generate_time_slots()


# ========== 3. 业务函数 ==========
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
    if 1080 <= t < 1260:  # 黄金时段票价上浮
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


# ========== 4. 检查排片所有硬约束 ==========
def is_feasible(schedule):
    # a. 每厅同一时间不重叠，每两场间隔≥15min
    for room in schedule:
        times = []
        for show in schedule[room]:
            t, m, v = show["time"], show["movie"], show["version"]
            runtime = movie_info[m]["runtime"]
            times.append((t, t + runtime))
        times.sort()
        for i in range(1, len(times)):
            if times[i][0] < times[i - 1][1] + 15:
                return False
    # b. 3D/IMAX总时长限制
    total_3d = total_imax = 0
    for room in schedule:
        for show in schedule[room]:
            m, v = show["movie"], show["version"]
            rt = movie_info[m]["runtime"]
            if v == "3D":
                total_3d += rt
            if v == "IMAX":
                total_imax += rt
    if total_3d > 1200 or total_imax > 1500:
        return False
    # c. 单厅每9小时≤7小时播放
    for room in schedule:
        shows = sorted(schedule[room], key=lambda x: x["time"])
        for i in range(len(shows)):
            start = shows[i]["time"]
            end = start + 540  # 9小时=540min
            total = 0
            for j in range(i, len(shows)):
                t2 = shows[j]["time"]
                m2 = shows[j]["movie"]
                if t2 >= end:
                    break
                total += movie_info[m2]["runtime"]
            if total > 420:  # 7小时=420min
                return False
    # d. 题材播放次数上下限
    genre_count = {}
    for room in schedule:
        for show in schedule[room]:
            m = show["movie"]
            genre = movie_info[m]["genres"]
            genre_count[genre] = genre_count.get(genre, 0) + 1
    genre_limit = {
        "Animation": (1, 5),
        "Horror": (0, 3),
        "Action": (2, 6),
        "Drama": (1, 6),
    }
    for g, (minv, maxv) in genre_limit.items():
        cnt = genre_count.get(g, 0)
        if cnt < minv or cnt > maxv:
            return False
    # e. 题材播放时间段限制
    for room in schedule:
        for show in schedule[room]:
            m = show["movie"]
            genre = movie_info[m]["genres"]
            t = show["time"]
            if genre in ["Animation", "Family"]:
                if t >= 1140:  # 19:00
                    return False
            if genre in ["Horror", "Thriller"]:
                if t < 1260:  # 21:00
                    return False
    return True


# ========== 5. 计算净收益 ==========
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


# ========== 6. 生成初始解（贪心，评分高优先） ==========
def generate_initial_solution():
    schedule = {room: [] for room in rooms}
    movies_sorted = sorted(movies, key=lambda m: rating_dict[m], reverse=True)
    for room in rooms:
        t = 600
        used = set()
        while t < 1800:
            m = None
            v = None
            for cand in movies_sorted:
                for vv in movie_info[cand]["versions"]:
                    if vv in room_info[room]["versions"] and cand not in used:
                        m, v = cand, vv
                        break
                if m:
                    break
            if m is None:
                break
            schedule[room].append({"time": t, "movie": m, "version": v})
            used.add(m)
            t += movie_info[m]["runtime"] + 15
    return schedule


# ========== 7. 邻域操作 ==========
def neighbor(schedule):
    new_schedule = copy.deepcopy(schedule)
    room = random.choice(rooms)
    if not new_schedule[room]:
        return new_schedule
    idx = random.randint(0, len(new_schedule[room]) - 1)
    old = new_schedule[room][idx]
    actions = ["movie", "version", "time"]
    action = random.choice(actions)
    if action == "movie":
        cands = []
        for m in movies:
            for v in movie_info[m]["versions"]:
                if v in room_info[room]["versions"]:
                    cands.append((m, v))
        m, v = random.choice(cands)
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


# ========== 8. 模拟退火主流程 ==========
def simulated_annealing(max_iter=1000, T0=1500, Tmin=1e-3, alpha=0.97):
    curr = generate_initial_solution()
    while not is_feasible(curr):
        curr = generate_initial_solution()
    curr_score = compute_objective(curr)
    best, best_score = copy.deepcopy(curr), curr_score
    T = T0
    pbar = tqdm(range(max_iter))
    for step in pbar:
        for _ in range(20):
            next_sol = neighbor(curr)
            if not is_feasible(next_sol):
                continue
            next_score = compute_objective(next_sol)
            dE = next_score - curr_score
            if dE > 0 or np.exp(dE / (T + 1e-6)) > random.random():
                curr, curr_score = next_sol, next_score
                if curr_score > best_score:
                    best, best_score = copy.deepcopy(curr), curr_score
        T *= alpha
        pbar.set_description(f"T={T:.2f}, BestNetProfit={best_score:.2f}")
        if T < Tmin:
            break
    return best, best_score


# ========== 9. 运行并输出 ==========
if __name__ == "__main__":
    best_schedule, best_score = simulated_annealing(
        max_iter=400, T0=1500, Tmin=0.5, alpha=0.97
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
    result_df.to_csv(output_path, index=False)
    print(f"已保存预测结果到 {output_path}")
    print(result_df)
