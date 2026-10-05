"""
This file contains the code that was used to determine the effect of
different epsilon values on the utilities. Most of the code is identical
to the code in val_iter.py
"""

from final_code import states, actions, transitions, utilities
import matplotlib.pyplot as plt

eps_val = [10e-1, 10e-2, 10e-3, 10e-4, 10e-5, 10e-6, 10e-7]
num_iters = []
utils = []

for e in eps_val:
    V = {state: 0.0 for state in states}
    policy = {state: None for state in states}
    epsilon = e
    iterations = 0

    while True:
        iterations += 1
        delta = 0
        new_V = V.copy()

        expected_utilities = []
        for state in states:
            state = state[2:] if state.startswith("A_") else state
            list_states = []
            if state in actions:
                best_states = []
                for action in actions[state]:
                    action = action[2:]
                    best_states.append((V[action], f"A_{action}"))
                list_states.append(max(best_states))

            else:
                if state == "Death":
                    list_states.append((0.0, "Death"))
                else:
                    for s, prob in transitions[state]:
                        value = prob * V.get(s, 0.0)
                        list_states.append((value, s))

            new_val, new_state = max(list_states)
            new_V[state] = utilities.get(state, 0.0) + new_val
            policy[state] = new_state

            diff = abs(new_V[state] - V[state])
            delta = diff if delta < diff else delta
        V = new_V

        if delta < epsilon:
            break

    num_iters.append(iterations)
    utils.append(
        (
            V["CKD5"],
            V["Dialysis"],
            V["CC"],
            V["Transplant"],
            V["Fistula"],
            V["OneMonth"],
        )
    )

print("utils", utils)
print("ites", num_iters)

plt.figure(figsize=(8, 5))
plt.plot(
    eps_val, num_iters, marker="s", linestyle="-", color="black", label="Iterations"
)
plt.xscale("log")
plt.gca().invert_xaxis() # so it goes from larger epsilon to smaller epsilon values
plt.xlabel("Epsilon")
plt.ylabel("Number of Iterations")
plt.title("Convergence Iterations vs. Epsilon")
plt.grid(True)
plt.legend()
plt.show()
