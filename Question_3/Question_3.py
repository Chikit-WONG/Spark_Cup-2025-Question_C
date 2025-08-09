import os
import math
import pandas as pd
from ortools.sat.python import cp_model

# ===== 0. 配置 =====
INPUT_CINEMA = r'C:\Users\Lenovo\Desktop\code\Spark_Cup-2025-Question_C-1\Data\input_data\df_cinema.csv'
INPUT_MOVIES = r'C:\Users\Lenovo\Desktop\code\Spark_Cup-2025-Question_C-1\Data\input_data\df_movies_schedule.csv'
OUT_PATH = r'C:\Users\Lenovo\Desktop\code\Spark_Cup-2025-Question_C-1\df_result_2.csv'
df_pred_rating = pd.read_csv(
    r"C:\Users\Lenovo\Desktop\code\Spark_Cup-2025-Question_C-1\Data\output_result\df_result_1.csv"
)

# 时间刻度：10:00 -> 次日03:00 共 17 小时 -> 17*60/15 = 68 个 15 分钟刻
START_MINUTE = 10 * 60
END_MINUTE = 27 * 60  # 03:00 next day = 27:00 in minutes
SLOT_MIN = 15
NUM_SLOTS = (END_MINUTE - START_MINUTE) // SLOT_MIN  # 68

# 题目常量
BASIC_COST_PER_SEAT = 2.42
FIXED_COST = 90
GOLDEN_START = 18 * 60
GOLDEN_END = 21 * 60
SHARE_MANDARIN = 0.43
SHARE_IMPORT = 0.51
THREE_D_TOTAL_LIMIT = 1200  # minutes
IMAX_TOTAL_LIMIT = 1500     # minutes

# 题材次数上下限
GENRE_LIMITS = {
    'Animation': (1, 5),
    'Horror': (0, 3),
    'Action': (2, 6),
    'Drama': (1, 6)
}

# 题材时间限制（start time constraints）
# Format: genre -> (earliest_start_minute or None, latest_start_minute or None)
GENRE_TIME_WINDOW = {
    'Animation': (None, 19 * 60),  # latest start 19:00
    'Family': (None, 19 * 60),
    'Horror': (21 * 60, None),     # earliest start 21:00
    'Thriller': (21 * 60, None)
}

# ===== 1. 读取数据 =====
print("读取数据...")
df_cinema = pd.read_csv(INPUT_CINEMA)
df_movies = pd.read_csv(INPUT_MOVIES)
# 读取预测评分（第二问输出）
df_pred_rating = pd.read_csv(
    r"C:\Users\Lenovo\Desktop\code\Spark_Cup-2025-Question_C-1\Data\output_result\df_result_1.csv"
)
# 合并预测评分到 df_movies（按 id 匹配）
df_movies = df_movies.drop(columns=['rating'], errors='ignore')
df_movies = df_movies.merge(df_pred_rating, on='id', how='left')

print(f"放映厅数量: {len(df_cinema)}, 待排片电影数: {len(df_movies)}")
print(f"时间刻数 NUM_SLOTS={NUM_SLOTS} (每刻 {SLOT_MIN} 分钟)")

# 预处理字典查表，减少 DataFrame 频繁访问
rooms = list(df_cinema['room'])
room_capacity = {r: int(df_cinema.loc[df_cinema.room == r, 'capacity'].values[0]) for r in rooms}
room_support_3d = {r: bool(df_cinema.loc[df_cinema.room == r, '3D'].values[0]) for r in rooms}
room_support_imax = {r: bool(df_cinema.loc[df_cinema.room == r, 'IMAX'].values[0]) for r in rooms}

movies = list(df_movies['id'])
# movie metadata dictionaries
movie_version = {}
movie_runtime = {}
movie_runtime_rounded = {}
movie_basic_price = {}
movie_rating = {}
movie_lang = {}
movie_genres = {}

for m in movies:
    row = df_movies[df_movies.id == m].iloc[0]
    movie_version[m] = row['version']
    runtime_min = int(row['runtime'])
    runtime_rounded = int(math.ceil(runtime_min / SLOT_MIN) * SLOT_MIN)
    movie_runtime[m] = runtime_min
    movie_runtime_rounded[m] = runtime_rounded
    movie_basic_price[m] = float(row['basic_price'])
    movie_rating[m] = float(row['rating'])  # 已替换为预测值
    movie_lang[m] = str(row['original_language'])
    movie_genres[m] = str(row['genres'])


# ===== 2. 模型初始化 =====
print("初始化模型...")
model = cp_model.CpModel()

# ===== 3. 仅为可行 (m,r,t) 创建变量（裁剪） =====
print("创建可行的决策变量 (裁剪不可行组合)...")
x = {}  # binary start decision: movie m starts in room r at slot t
# We'll also create play/busy occupancy variables per room-slot
y_play = {}  # room r is playing movie at slot s (true if any movie plays in that slot)
y_busy = {}  # room r is busy at slot s (playing or cleaning)
for r in rooms:
    for s in range(NUM_SLOTS):
        y_play[(r, s)] = model.NewBoolVar(f"yplay_{r}_{s}")
        y_busy[(r, s)] = model.NewBoolVar(f"ybusy_{r}_{s}")

# generate feasible start variables
feasible_count = 0
for m in movies:
    version = movie_version[m]
    runtime_min = movie_runtime_rounded[m]  # minutes rounded to nearest slot multiple
    dur_slots = runtime_min // SLOT_MIN
    for r in rooms:
        # version support
        if version == '3D' and not room_support_3d[r]:
            continue
        if version == 'IMAX' and not room_support_imax[r]:
            continue
        for t in range(NUM_SLOTS):
            start_minute = START_MINUTE + t * SLOT_MIN
            end_minute = start_minute + runtime_min
            # must end before or at END_MINUTE
            if end_minute > END_MINUTE:
                continue
            # genre time window check
            genres = movie_genres[m]
            violates_time = False
            for g, (earliest, latest) in GENRE_TIME_WINDOW.items():
                if g in genres:
                    if earliest is not None and start_minute < earliest:
                        violates_time = True
                        break
                    if latest is not None and start_minute > latest:
                        violates_time = True
                        break
            if violates_time:
                continue
            # passed basic feasibility -> create var
            x[(m, r, t)] = model.NewBoolVar(f"x_{m}_{r}_{t}")
            feasible_count += 1

print(f"可行候选 (m,r,t) 变量数: {feasible_count}")

# If no feasible candidates, exit
if feasible_count == 0:
    print("没有可行的 (m,r,t) 变量，退出。请检查输入数据或时间/版本约束。")
    exit(0)

# ===== 4. 预计算每候选场次的净收益（profit） =====
print("预计算 profit（净收益）...")
profit = {}
for (m, r, t), var in x.items():
    version = movie_version[m]
    cap = room_capacity[r]
    rating = movie_rating[m]
    attendance = int(math.floor(cap * rating / 10.0))  # 向下取整
    base_price = movie_basic_price[m]
    price = base_price * (1.2 if version == '3D' else 1.23 if version == 'IMAX' else 1.0)
    show_time_min = START_MINUTE + t * SLOT_MIN
    if GOLDEN_START <= show_time_min < GOLDEN_END:
        price *= 1.3
    revenue = attendance * price
    share = SHARE_MANDARIN if ('Mandarin' in movie_lang[m]) else SHARE_IMPORT
    net_income = revenue * (1 - share)
    cap_cost = BASIC_COST_PER_SEAT * cap
    version_coef = 1.0 if version == '2D' else (1.1 if version == '3D' else 1.15)
    cost = version_coef * cap_cost + FIXED_COST
    profit[(m, r, t)] = int(round(net_income - cost, 2))

# ===== 5. 目标函数 =====
print("建立目标函数（最大化净收益）...")
objective_terms = []
for key, var in x.items():
    objective_terms.append(profit[key] * var)
model.Maximize(sum(objective_terms))

# ===== 6. 约束 =====

print("添加不重叠、占用与清理间隔约束...")

# Link start x -> y_play and y_busy (playing slots and busy slots with cleaning)
for (m, r, t), var in x.items():
    runtime_min = movie_runtime_rounded[m]
    dur_slots = runtime_min // SLOT_MIN
    # playing occupies slots [t, t + dur_slots - 1]
    for s in range(t, t + dur_slots):
        if s < NUM_SLOTS:
            # if start var is 1 -> that slot is playing
            model.Add(y_play[(r, s)] >= var)
            model.Add(y_busy[(r, s)] >= var)
    # cleaning occupies next slot t + dur_slots (15 minutes)
    clean_slot = t + dur_slots
    if clean_slot < NUM_SLOTS:
        model.Add(y_busy[(r, clean_slot)] >= var)

# At most one playing per room per slot
for r in rooms:
    for s in range(NUM_SLOTS):
        model.Add(sum(y_play[(r, s2)] for s2 in [s]) <= 1)  # trivial but keeps form
        # At most one busy per slot (no overlapping play or cleaning)
        model.Add(sum(y_busy[(r, s2)] for s2 in [s]) <= 1)

# The above trivial per-slot constraints need to be replaced by aggregated constraints:
# Enforce that for each room and slot, sum of starts that occupy that slot <= 1.
# (This avoids double-booking via starts rather than via y vars.)
for r in rooms:
    for s in range(NUM_SLOTS):
        occupying_starts = []
        for (m2, r2, t2), var2 in x.items():
            if r2 != r:
                continue
            runtime_min2 = movie_runtime_rounded[m2]
            dur_slots2 = runtime_min2 // SLOT_MIN
            # if a start at t2 would play during slot s or clean at slot s
            if t2 <= s <= t2 + dur_slots2:  # includes cleaning slot at t2+dur_slots2
                occupying_starts.append(var2)
        if occupying_starts:
            model.Add(sum(occupying_starts) <= 1)

print("添加 9 小时滑动窗口内播放时长 ≤ 7 小时 约束（单位为刻）...")
# 9小时窗口 = 9*60 / 15 = 36 slots; 7小时播放上限 = 7*60 /15 = 28 slots
WINDOW_SLOTS = (9 * 60) // SLOT_MIN
MAX_PLAY_SLOTS = (7 * 60) // SLOT_MIN
for r in rooms:
    for start_slot in range(0, NUM_SLOTS):
        end_slot = start_slot + WINDOW_SLOTS
        if end_slot > NUM_SLOTS:
            break
        # compute sum of playing slots in [start_slot, end_slot-1] <= MAX_PLAY_SLOTS
        # To do this, we sum y_play[r,s] for s in window. But we have y_play linked via starts.
        model.Add(sum(y_play[(r, s)] for s in range(start_slot, end_slot)) <= MAX_PLAY_SLOTS)

print("添加题材播放次数上下限约束...")
# For counting showings per genre, we count starts x[(m,r,t)]
for genre, (gmin, gmax) in GENRE_LIMITS.items():
    # sum starts for movies whose genres contain the genre keyword
    starts = []
    for (m, r, t), var in x.items():
        if genre in movie_genres[m]:
            starts.append(var)
    if starts:
        if gmin is not None:
            model.Add(sum(starts) >= gmin)
        if gmax is not None:
            model.Add(sum(starts) <= gmax)
    else:
        # if no movies of that genre exist in candidates, we may skip or ensure gmin==0; for this dataset we skip
        pass

print("添加 3D/IMAX 全影院总播放时长上限约束...")
three_d_terms = []
imax_terms = []
for (m, r, t), var in x.items():
    ver = movie_version[m]
    runtime_min = movie_runtime[m]  # original runtime minutes (not rounded)
    if ver == '3D':
        three_d_terms.append(runtime_min * var)
    if ver == 'IMAX':
        imax_terms.append(runtime_min * var)
if three_d_terms:
    model.Add(sum(three_d_terms) <= THREE_D_TOTAL_LIMIT)
if imax_terms:
    model.Add(sum(imax_terms) <= IMAX_TOTAL_LIMIT)

print("保证每场电影开始时间为整刻（已隐含）并且结束在凌晨3点前（已在可行性生成时检查）")

# 题材时间窗已在生成可行变量时排除，因此无需再次添加

# ===== 7. 求解 =====
print("开始求解...（注意：若数据规模大，求解可能耗时）")
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60  # 调试时可短，正式运行可放大或移除
solver.parameters.num_search_workers = 8
solver.parameters.log_search_progress = True

status = solver.Solve(model)

print(f"Solve status: {solver.StatusName(status)}")
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(f"Objective value: {solver.ObjectiveValue()}")
else:
    print("未找到可行解或求解器未返回解。")

# ===== 8. 提取并输出结果 =====
schedules = []
for (m, r, t), var in x.items():
    if solver.Value(var) == 1:
        start_min = START_MINUTE + t * SLOT_MIN
        hh = start_min // 60
        mm = start_min % 60
        showtime_str = f"{hh:02d}:{mm:02d}"
        attendance = int(math.floor(room_capacity[r] * movie_rating[m] / 10.0))
        schedules.append({
            'room': r,
            'showtime': showtime_str,
            'id': m,
            'version': movie_version[m],
            'attendance': attendance
        })

df_result = pd.DataFrame(schedules, columns=['room', 'showtime', 'id', 'version', 'attendance'])
df_result.to_csv(OUT_PATH, index=False)
print(f"排片表已保存，共 {len(schedules)} 场，路径: {OUT_PATH}")
