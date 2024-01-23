from spx_1d_lstm_model_trainer import Spx1dLstmModelTrainer

def run():
    #TODO : Load Model Trainers and their hyper parameters from config file
    sliding_window_lengths = [None, 75, 100, 150, 200, 250, 500, 750]
    sequence_lengths = [3, 5, 7, 10, 15, 20, 25, 35, 50, 65, 80, 100, 125]
    hidden_sizes = [1, 2, 3, 6, 16]
    fc_neurons = [16,128,1024]
    layers = [1,2,3]

    while True:
        for sw in sliding_window_lengths[1:]:
            for seq in sequence_lengths:
                for hidden_size in hidden_sizes:
                    for fc_neuron in fc_neurons:
                        for num_layers in layers:

                            params = {
                            "seq_len": seq,
                            "hidden_size": hidden_size,
                            "num_layers": num_layers,
                            "fc_neurons": fc_neuron,
                            "n_epochs": 200,
                            "learning_rate" : 0.001,
                            "sliding_window" : sw
                            }
                            
                            trainer = Spx1dLstmModelTrainer()
                            trainer.update_data_and_experiment()
                            trainer.update_hyperparams(**params)
                            trainer.run()


                            



if __name__=="__main__":
    run()
