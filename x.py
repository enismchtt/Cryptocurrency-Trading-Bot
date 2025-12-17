import itertools
from src import config

# The feature that MUST be in every list

all_combinations = []

# Generate combinations of length 0 up to length 6
for r in range(len(config.input_types) + 1):
    # itertools.combinations returns tuples, so we convert to list
    for combo in itertools.combinations(config.input_types, r):
        # Combine the optional parts with the mandatory part
        full_combo = list(combo) + [config.pred]
        all_combinations.append(full_combo)

for i in range(len(all_combinations)):
    print(all_combinations[i])