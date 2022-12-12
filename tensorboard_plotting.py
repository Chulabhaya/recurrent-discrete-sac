import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import seaborn as sns

log_dirs = ["runs/paper_results/100percent_expert_POMDP-heavenhell_1-episodic-v0__sac_discrete_obs_discrete_action_recurrent_offline__1__1670535594",
            "runs/paper_results/25_percent_random_POMDP-heavenhell_1-episodic-v0__sac_discrete_obs_discrete_action_recurrent_offline__1__1670546209",
            "runs/paper_results/50_percent_random_POMDP-heavenhell_1-episodic-v0__sac_discrete_obs_discrete_action_recurrent_offline__1__1670554456",
            "runs/paper_results/75_percent_random_POMDP-heavenhell_1-episodic-v0__sac_discrete_obs_discrete_action_recurrent_offline__1__1670594858",
            "runs/paper_results/100_percent_random_POMDP-heavenhell_1-episodic-v0__sac_discrete_obs_discrete_action_recurrent_offline__1__1670612893",
            ]
log_names = ["0% Random (Expert)",
             "25% Random",
             "50% Random",
             "75% Random",
             "100% Random"]
smoothed_log_names = ["0% Random (Expert)",
             "25% Random",
             "50% Random",
             "75% Random",
             "100% Random"]
df_list = []
smoothed_df_list = []
df_smoothing_weight = 0.99

for i, log_dir in enumerate(log_dirs):
    chart_scalar = "charts/eval_episodic_return"
    y_key = "Episodic Return"
    x_key = "Timesteps"
    name = log_dir

    event_accumulator = EventAccumulator(log_dir)
    event_accumulator.Reload()

    events = event_accumulator.Scalars(chart_scalar)
    x = [x.step for x in events]
    y = [x.value for x in events]

    df = pd.DataFrame({"name": log_names[i], x_key: x, y_key: y})
    df_list.append(df)

    smooth_df = df.ewm(alpha=(1 - df_smoothing_weight)).mean()
    smoothed_df_list.append(smooth_df)

fig, ax = plt.subplots()
line_colors = []
for i, df in enumerate(df_list):
    sns.lineplot(data=df, x=x_key, y=y_key, ax=ax, alpha=0.4, label=log_names[i])
    line_color = ax.lines[-1].get_color()
    line_colors.append(line_color)
# for i, df in enumerate(smoothed_df_list):
#     sns.lineplot(data=df, x=x_key, y=y_key, label=smoothed_log_names[i], ax=ax)
plt.grid(alpha=0.3)
plt.legend(loc='lower right')
plt.title("Offline Reinforcement Learning Dataset Mixing Comparison")
fig.savefig("offline_rl.png", dpi=300)
