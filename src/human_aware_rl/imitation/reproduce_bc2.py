import os
from human_aware_rl.imitation.behavior_cloning_tf22 import (
    get_bc_params,
    train_bc_model,
)
from human_aware_rl.static import (
    CLEAN_2019_HUMAN_DATA_TEST,
    CLEAN_2019_HUMAN_DATA_TRAIN,
)

if __name__ == "__main__":
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    bc_dir = os.path.join(current_file_dir, "bc_runs", "train", "multi_layout")
    
    if not os.path.isdir(bc_dir):
        # Define parameters for training on all layouts
        params_to_override = {
            "layouts": [
                "random3",
                "coordination_ring",
                "cramped_room",
                "random0",
                "asymmetric_advantages",
            ],
            "data_path": CLEAN_2019_HUMAN_DATA_TRAIN,
            "epochs": 100,
            "old_dynamics": True,
        }
        # Get combined BC parameters
        bc_params = get_bc_params(**params_to_override)
        
        # Train the model on all layouts combined
        train_bc_model(bc_dir, bc_params, True)
    else:
        print(f"Model directory {bc_dir} already exists. Skipping training.")