# Base string
base_path = "src/human_aware_rl/imitation/bc_runs/train/"

# List of replacements
replacements = [
    "random3",
    "coordination_ring",
    "cramped_room",
    "random0",
    "asymmetric_advantages"
]

# Loop to append each replacement to the base path
for replacement in replacements:
    modified_string = f"{base_path}{replacement}"
    print(modified_string)  # Or use the modified string for other purposes