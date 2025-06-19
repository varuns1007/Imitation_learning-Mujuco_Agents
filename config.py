
'''
This file contains the configs used for Model creation and training. You need to give your best hyperparameters and the configs you used to get the best results for 
every environment and experiment.  These configs will be automatically loaded and used to create and train your model in our servers.
'''
#You can add extra keys or modify to the values of the existing keys in bottom level of the dictionary.
#DO NOT CHANGE THE STRUCTURE OF THE DICTIONARY. 

configs = {
    
    'Hopper-v4': {
            #You can add or change the keys here
              "hyperparameters": {
                'hidden_size': 32,
                'n_layers': 2,
                'batch_size': 512, 
                'learning_rate': 1e-3,
                'activation': 'leaky_relu',
                'save': True,
                'buffer_size': 10000,
            },
            "num_iteration": 300,
            "episode_len":2000
    },
    
    
    'Ant-v4': {
            #You can add or change the keys here
              "hyperparameters": {
                'hidden_size': 32,
                'n_layers': 2,
                'batch_size': 512, 
                'learning_rate': 1e-3,
                'activation': 'leaky_relu',
                'save': True,
                'buffer_size': 100000,
            },
            "num_iteration": 460,
            # "episode_len":1500

    }

}