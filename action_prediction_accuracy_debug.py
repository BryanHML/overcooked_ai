import numpy as np
import pandas as pd
import json

from human_aware_rl.imitation.behavior_cloning_tf2 import BehaviorCloningPolicy, load_bc_model
from human_aware_rl.rllib.rllib import get_base_ae
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, Recipe
from scipy.special import softmax
from human_aware_rl.human.process_dataframes import get_human_human_trajectories
from human_aware_rl.imitation.behavior_cloning_tf2 import _get_base_ae, BehaviorCloningPolicy
from human_aware_rl.rllib.rllib import RlLibAgent 

#trajectory dict_keys(['metadatas', 'ep_states', 'ep_actions', 'ep_rewards', 'ep_infos', 'ep_dones', 'env_params', 'ep_returns', 'ep_lengths', 'mdp_params'])

# Load model
model_path = "src/human_aware_rl/imitation/bc_runs/train/"

# Load the dataset
dataset_path = "src/human_aware_rl/static/human_data/cleaned/2019_hh_trials_train.pickle"

#Select train or test
train_or_test = "train"


def compute_action_prediction_accuracy(bc_agent, trajectories):
    """
    Computes the action prediction accuracy of an agent of every layout.

    Args:
        bc_agent: The agent with a .action() method that predicts actions given an OvercookedState.
        trajectories: Dictionary containing trajectory data with keys 'ep_states' and 'ep_actions'.

    Returns:
        accuracy: The percentage of correctly predicted actions.
    """
    # Extract the states and actions from the trajectories
    
    ep_states = trajectories['ep_states']  # List of lists of OvercookedState objects
    ep_actions = trajectories['ep_actions']  # List of lists of tuples representing actual actions

    total_actions = 0
    correct_predictions = 0

    

    # Iterate over each episode
    for list_of_states, list_of_actions in zip(ep_states, ep_actions):
        # Ensure state and action lists have the same length
        assert len(list_of_states) == len(list_of_actions), "Mismatch in states and actions length"

        # Iterate over states and corresponding actions
        for state, actual_action in zip(list_of_states, list_of_actions):
            
            # Predict the action using the agent
            predicted_action, _ = bc_agent.action(state)

            # Compare predicted action with actual action
            if predicted_action == actual_action:
                correct_predictions += 1

            total_actions += 1

    # Compute accuracy as a percentage
    accuracy = (correct_predictions / total_actions) * 100 if total_actions > 0 else 0.0

    return accuracy



def predict_accuracy_all_layout(model_path, dataset_path, train_or_test):
    """
    Predicts the action accuracy of an layout-specific agent for every layout in overcooked_ai

    Args:
        model_path: Path to find trained agent.
        dataset_path: Path to find dataset to predict action accuracy.
        train_or_test: String either "train" or "test" to correctly identify trajectories.

    Returns:
        action_predict_accuracy_dict: Dictionary containing action prediction accuracy of each layout and specified set used.
    """

    #Initialise empty dictionary to store action prediction accuracy of each layout 
    action_predict_accuracy_dict = {}

    #Append which set is used 
    action_predict_accuracy_dict["Which Set?"] = train_or_test

    # List of layouts for overcooked_ai
    # layouts = [
    #     "random3",
    #     "coordination_ring",
    #     "cramped_room",
    #     "random0",
    #     "asymmetric_advantages"
    #     ]

    # DEBUG with one layout
    layouts = ["cramped_room"]

    # Loop to append each layout to the model_path
    for layout in layouts:
        
        #Create variable that identifies path to specific layout agent
        specific_agent_layout_path = f"{model_path}{layout}"

        #Load trained model and its parameters 
        model, bc_params = load_bc_model(specific_agent_layout_path, verbose=True)

        # Create the BC policy object
        policy = BehaviorCloningPolicy.from_model(model, bc_params, stochastic=True)

        # Initialize the Overcooked environment
        base_ae = get_base_ae(bc_params["mdp_params"], bc_params["env_params"])


        # {'layout_name': 'cramped_room', 'old_dynamics': True}
        print("mdp_params:", bc_params["mdp_params"])

        # {'horizon': 400, 'mlam_params': {'start_orientations': False, 'wait_allowed': False, 'counter_goals': [], 'counter_drop': [], 'counter_pickup': [], 'same_motion_goals': True}}
        print("env_params:", bc_params["env_params"])

        #<overcooked_ai_py.agents.benchmarking.AgentEvaluator object at 0x0000023FB94F14C8>
        # print("base_ae:", base_ae)



        base_env = base_ae.env

        #Wrap loaded model as RllibAgent
        compute_agent = RlLibAgent(policy,0,base_env.featurize_state_mdp)

        # Initialize rnn_state
        compute_agent.rnn_state = policy.get_initial_state()  

        #Get trajectories for specific layout where states are overcookedstate objs
        trajectories_compute = get_human_human_trajectories(
            layouts=[layout],
            dataset_type = train_or_test,
            data_path = dataset_path,
            featurize_states=False #OvercookedState Object if False
            )
        
        # the ep states key in trajectories contains list of overcookedstates 
        # print("states", trajectories_compute["ep_states"][0][0])

        #state looks like this 
        #Players: ((1, 2) facing (0, -1) holding None, (3, 1) facing (0, -1) holding None), Objects: [], Bonus orders: [] All orders: [('onion',), ('onion', 'onion'), ('onion', 'onion', 'onion'), ('tomato',), 
        # ('tomato', 'tomato'), ('tomato', 'tomato', 'tomato'), ('onion', 'tomato'), ('onion', 'onion', 'tomato'), ('onion', 'tomato', 'tomato')] Timestep: 0

        # #Compute action prediction accuracy for specific layout
        # current_layout_accuracy = compute_action_prediction_accuracy(compute_agent, trajectories_compute)

        # #Round to 2dp and add percentage
        # formatted_accuracy = f"{current_layout_accuracy:.2f}%"
        # action_predict_accuracy_dict[layout] = formatted_accuracy

    return action_predict_accuracy_dict

print(predict_accuracy_all_layout(model_path, dataset_path, train_or_test))