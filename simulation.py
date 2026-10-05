"""
This file contains the code for the Monte Carlo simulation.
"""

from final_code import states, actions, transitions, utilities
import random
import pandas as pd
import matplotlib.pyplot as plt

num_individuals = 100000
start_state = "CKD5"
results = []

for i in range(num_individuals):
    current_state = start_state
    months = 0
    choices = []
    Qol = 0.0
    while current_state != "Death":
        if current_state in actions:
            current_state = random.choice(actions[current_state])[2:]
            choices.append(current_state)

        else:
            previous_state = current_state
            next_states, next_probs = list(zip(*transitions[current_state]))
            current_state = random.choices(next_states, weights=next_probs, k=1)[0]
            if current_state in utilities:
                Qol += utilities[current_state]
                months += 1
    results.append((choices, months, Qol / months))

df = pd.DataFrame(results, columns=["actions", "months", "QoL"])

df["action_combined"] = df["actions"].apply(lambda x: ", ".join(x))
grouped_stats = df.groupby("action_combined")[["months", "QoL"]].agg(["mean", "std"])
grouped_stats["QALY"] = (
    grouped_stats[("months", "mean")] * grouped_stats[("QoL", "mean")] / 12
)
print(grouped_stats)
print(grouped_stats["QALY"])

# plots
actions = grouped_stats.index
qaly_values = grouped_stats[("QALY")].values
months_values = grouped_stats[("months", "mean")].values
qol_values = grouped_stats[("QoL", "mean")].values

plt.figure(figsize=(10, 6))
plt.bar(actions, qaly_values, color="skyblue")
plt.xlabel("Actions", fontsize=12)
plt.ylabel("QALY", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.show()
plt.figure(figsize=(10, 6))
for i in range(len(actions)):
    plt.plot([0, months_values[i]], [0, qol_values[i]], label=actions[i], marker="o")
plt.xlabel("Months (Mean)", fontsize=12)
plt.ylabel("QoL (Mean)", fontsize=12)
plt.legend()
plt.show()
