"""
This file contains all the code used to compute the value iteration
and to get the optimal policy.
"""

from final_code import states, actions, transitions, utilities

V = {state: 0.0 for state in states}
policy = {state: None for state in states}
epsilon = 1e-4
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

utils = [
    V["CKD5"],
    V["Dialysis"],
    V["CC"],
    V["Transplant"],
    V["Fistula"],
    V["OneMonth"],
]


print("V", V)
print("Policy", policy)
print("iterations", iterations)

print("_________________")
print("CKD5", "Dialysis", "CC", "Transplant", "Fistula", "OneMonth")
print(utils)

max_val = max(utils)
normalized_utils = [v / max_val for v in utils]
print(normalized_utils)

target_max = 0.887
max_val = max(utils)
normalized_utils = [v / max_val * target_max for v in utils]
print(normalized_utils)
