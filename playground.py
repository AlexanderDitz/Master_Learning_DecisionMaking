import json

my_dict = {
    'rnn_modules': ['x_learning_rate_reward', 'x_value_reward_not_chosen', 'x_value_choice_chosen', 'x_value_choice_not_chosen'],
    'control_parameters': ['c_action', 'c_reward', 'c_value_reward'],
    'sindy_library_config': {
        'x_learning_rate_reward': ['c_reward', 'c_value_reward'],
    },
    'sindy_filter_config': {
        'x_learning_rate_reward': ['c_action', 1, True],
        'x_value_reward_not_chosen': ['c_action', 0, True],
        'x_value_choice_chosen': ['c_action', 1, True],
        'x_value_choice_not_chosen': ['c_action', 0, True],
    },
}

with open("data.json", "w") as outfile:
    json.dump(my_dict, outfile, indent=4)
    
    
with open("data.json", "r") as infile:
    my_dict_2 = json.load(infile)
    
print(my_dict_2['sindy_library_config'])