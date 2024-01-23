from abc import ABC, abstractmethod
import torch
import numpy as np
import mlflow

class ModelTrainer(ABC):
    def __init__(self):
        self.pred_type = 'bc'
        self._model_1 = None
        self._model_2 = None
        self._data = None
        self._hyperparams = None
        self._device_1 = self.set_device()
        self._device_2 = self.set_device(gpu=1)
        self._experiment = None
        self._last_trainable_trade_date = None
        self.CONFIG = {
            'data_directory':r'/home/epsilonoptima/repos/cuda prototype/',
            'partial_filename':'spx_data',
            'reverse_data_order':True,
            'set_date_index':False,
            'mlflow_uri':"sqlite:////home/epsilonoptima/mlflow/mlruns.db",
            'artifact_directory': '////home/epsilonoptima/mlflow/artifacts',
            'temp_directory':'////home/epsilonoptima/mlflow/temp/cuda0',
         }

    def set_device(self, gpu=0):
        if torch.cuda.is_available(): 
            dev = f"cuda:{gpu}" 
        else: 
            dev = "cpu" 
        print(f"Device : {dev}")
        return torch.device(dev) 

    @property
    def data(self):
        return self._data
    
    def update_data(self, **kwargs):
        self._data = self._data

    @property
    def hyperparams(self):
        return self._hyperparams
    
    
    def update_hyperparams(self, **kwargs):
        self._hyperparams = kwargs
    
    @property
    def model(self):
        return self._model_1 
    
    def update_model(self, **kwargs):
        pass

    @property
    def experiment(self):
        return self._experiment
    
    def update_experiment(self):
        set_experiment = True
        
        if self._experiment is not None:
            if self._last_trainable_trade_date in self._experiment.name:
                set_experiment = False

        if set_experiment:
            # Set up ML Flow
            mlflow.set_tracking_uri(self.CONFIG['mlflow_uri'])
            # Sets the current active experiment to the "Apple_Models" experiment and
            # returns the Experiment metadata
            curr_experiment = mlflow.set_experiment(f"{self.__class__.__name__}_{self._last_trainable_trade_date}")
            self._experiment = curr_experiment


    def update_data_and_experiment(self):
        self.update_data()
        self.update_experiment()

    
    @abstractmethod
    def run(self):
        pass


    def get_start_end_idx(self,test_set_len=252):
        return self._data.shape[0]-test_set_len, self._data.shape[0]

    def make_tensors(self,ss, data, seq_len, device):
        X = data.iloc[:,:-1].values
        y = data.iloc[:,-1:].values

        X_trans = ss.fit_transform(X)

        X_ss, y_seq = self.split_sequences(X_trans, y, seq_len, 1)

        # To simulate real market inference conditions, we leave a 1 day gap between training and inference
        X_train = X_ss[:-2]
        y_train = y_seq[:-2]

        X_test = X_ss[-1:]
        y_test = y_seq[-1:]

        X_train_tensors = torch.Tensor(X_train).to(device)
        y_train_tensors = torch.Tensor(y_train).to(device)

        # Tensor dimensions (batch size, sequence, features)
        X_train_tensors = torch.reshape(X_train_tensors,   
                                        (X_train_tensors.shape[0], seq_len, 
                                        X_train_tensors.shape[2]))

        X_test_tensors = torch.Tensor(X_test).to(device)
        y_test_tensors = torch.Tensor(y_test).to(device)

        X_test_tensors = torch.reshape(X_test_tensors,   
                                        (X_test_tensors.shape[0], seq_len, 
                                        X_test_tensors.shape[2]))
        
        return X_train_tensors, y_train_tensors, X_test_tensors, y_test_tensors

    def make_cross_val_df(self,idx,sw):
        if sw is None:
            return self.data.iloc[:idx]
        else:
            return self.data.iloc[idx-sw:idx]

    # split a multivariate sequence past, future samples (X and y)
    def split_sequences(self, input_sequences, output_sequence, n_steps_in, n_steps_out, n_step_gap = 1):
        X, y = list(), list() # instantiate X and y
        input_rows = input_sequences.shape[0]
        for i in range(input_rows):
            # find the end of the input, output sequence
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out - 1
            # check if we are beyond the dataset
            if out_end_ix > len(input_sequences): break
            # gather input and output of the pattern
            seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
            X.append(seq_x), y.append(seq_y)
        return np.array(X), np.array(y)
    




