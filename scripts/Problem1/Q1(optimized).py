import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from pathlib import Path

# Load the CSV file
root_path = Path(__file__).parents[2]
file_path = root_path / 'data' /'Wimbledon_featured_matches.csv'
df = pd.read_csv(file_path)

class Match:
    def __init__(self, match_id, player1, player2, elapsed_time, set_no, game_no, point_no, p1_sets, p2_sets, p1_games, p2_games, p1_score, p2_score, server, serve_no, point_victor, p1_points_won, p2_points_won, game_victor, set_victor, p1_ace, p2_ace, p1_winner, p2_winner, winner_shot_type, p1_double_fault, p2_double_fault, p1_unf_err, p2_unf_err, p1_net_pt, p2_net_pt, p1_net_pt_won, p2_net_pt_won, p1_break_pt, p2_break_pt, p1_break_pt_won, p2_break_pt_won, p1_break_pt_missed, p2_break_pt_missed, p1_distance_run, p2_distance_run, rally_count, speed_mph, serve_width, serve_depth, return_depth):
        self.match_id = match_id
        self.player1 = player1
        self.player2 = player2
        self.elapsed_time = elapsed_time
        self.set_no = set_no
        self.game_no = game_no
        self.point_no = point_no
        self.p1_sets = p1_sets
        self.p2_sets = p2_sets
        self.p1_games = p1_games
        self.p2_games = p2_games
        self.p1_score = p1_score
        self.p2_score = p2_score
        self.server = server
        self.serve_no = serve_no
        self.point_victor = point_victor
        self.p1_points_won = p1_points_won
        self.p2_points_won = p2_points_won
        self.game_victor = game_victor
        self.set_victor = set_victor
        self.p1_ace = p1_ace
        self.p2_ace = p2_ace
        self.p1_winner = p1_winner
        self.p2_winner = p2_winner
        self.winner_shot_type = winner_shot_type
        self.p1_double_fault = p1_double_fault
        self.p2_double_fault = p2_double_fault
        self.p1_unf_err = p1_unf_err
        self.p2_unf_err = p2_unf_err
        self.p1_net_pt = p1_net_pt
        self.p2_net_pt = p2_net_pt
        self.p1_net_pt_won = p1_net_pt_won
        self.p2_net_pt_won = p2_net_pt_won
        self.p1_break_pt = p1_break_pt
        self.p2_break_pt = p2_break_pt
        self.p1_break_pt_won = p1_break_pt_won
        self.p2_break_pt_won = p2_break_pt_won
        self.p1_break_pt_missed = p1_break_pt_missed
        self.p2_break_pt_missed = p2_break_pt_missed
        self.p1_distance_run = p1_distance_run
        self.p2_distance_run = p2_distance_run
        self.rally_count = rally_count
        self.speed_mph = speed_mph
        self.serve_width = serve_width
        self.serve_depth = serve_depth
        self.return_depth = return_depth
        # 转换elapsed_time为秒
        self.elapsed_time_seconds = [self.convert_time_to_seconds(time_str) for time_str in self.elapsed_time]

    @staticmethod
    def convert_time_to_seconds(time_str):
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s


    def calculate_difference_value(self, current_server, current_point_victor, length):
        player1_server = player2_server = 0
        player1_server_points_won = player2_server_points_won = 0
        for i in range(length):
            if current_server[i] == 1:
                player1_server += 1
                if current_point_victor[i] == 1:
                    player1_server_points_won += 1
            elif current_server[i] == 2:
                player2_server += 1
                if current_point_victor[i] == 2:
                    player2_server_points_won += 1
        if player1_server > 0 and player2_server > 0:
            difference_value = (player1_server_points_won / player1_server - player2_server_points_won / player2_server + 1) / 2.0
        else:
            difference_value = 0.5  # Default value
        return difference_value

    def evaluation_performance(self, window_length=300, window_moving_length=60):
        start_time = min(self.elapsed_time_seconds)
        end_time = max(self.elapsed_time_seconds)
        difference_value_list = []
        selected_times = []

        current_time = start_time
        while current_time + window_length <= end_time:
            indices = [i for i, t in enumerate(self.elapsed_time_seconds) if current_time <= t < current_time + window_length]
            if indices:
                current_server = [self.server[i] for i in indices]
                current_point_victor = [self.point_victor[i] for i in indices]
                if len(current_server) > 0 and len(current_point_victor) > 0:
                    difference_value = self.calculate_difference_value(current_server, current_point_victor, len(indices))
                    difference_value_list.append(difference_value)
                    selected_times.append(current_time + window_length / 2)
            current_time += window_moving_length

        selected_times_hms = [Match.seconds_to_hms(time) for time in selected_times]
        return difference_value_list, selected_times, selected_times_hms


    def seconds_to_hms(seconds):
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

# 根据match_id创建Match实例列表
matches_list = [Match(*[group[col].values for col in group]) for match_id, group in df.groupby('match_id')]

# 选择要分析的比赛实例，这里以列表中的第一个比赛为例
match_instance = matches_list[0]

# 参数组合
params_combinations = [(450, 60), (600, 60), (600, 90)]

# 绘图
# 确定整个时间范围
start_time = min(match_instance.elapsed_time_seconds)
end_time = max(match_instance.elapsed_time_seconds)

# 生成每半小时的标记点，从0:30:00开始，直到比赛结束
tick_marks_seconds = np.arange(start_time, end_time, 1800) # 每半小时为1800秒
tick_labels = [Match.seconds_to_hms(tick) for tick in tick_marks_seconds]

# 画图
plt.figure(figsize=(10, 6))
for window_length, window_moving_length in params_combinations:
    difference_value_list, selected_times, _ = match_instance.evaluation_performance(window_length, window_moving_length)
    if difference_value_list:
        x = np.array(selected_times)  # 使用数值型的selected_times进行插值
        y = np.array(difference_value_list)
        f_cubic = interp1d(x, y, kind='cubic', fill_value="extrapolate")
        x_new = np.linspace(min(x), max(x), num=1000, endpoint=True)
        y_cubic = f_cubic(x_new)
        plt.plot(x_new, y_cubic, '-', label=f'Window: {window_length}s, Step: {window_moving_length}s')

# 设置自定义的X轴标签
plt.xticks(tick_marks_seconds, labels=tick_labels, rotation=45)

plt.axhline(y=0.5, color='r', linestyle='--', label='Equilibrium Line')
plt.xlabel('Elapsed Time (HH:MM:SS)')
plt.ylabel('Probability Difference')
plt.legend(loc='upper left')
plt.tight_layout()  # 调整布局以防止标签重叠
plt.show()
