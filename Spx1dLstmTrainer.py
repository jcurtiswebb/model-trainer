from ModelTrainer import ModelTrainer
from ModelStore import LSTM1
from utils.YfinanceUtils import *
from utils.ModelTrainerUtils import *


class Spx1dLstmModelTrainer(ModelTrainer):
    def __init__(self):
        super().__init__()

    def update_model(self, **kwargs):
        input_size = kwargs['input_size']
        hidden_size = kwargs['hidden_size']
        num_layers = kwargs['num_layers']
        fc_neurons = kwargs['fc_neurons']
        self._model = LSTM1(input_size, hidden_size, num_layers, fc_neurons, self._device)

    def update_data(self, **kwargs):
        df = get_latest_data()
        col = df.pop('Close')
        df.insert(df.shape[1], 'Close', col)

        df['Tomorrow_Close'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        # Turn tomorrow close into a binary classification
        df['Tomorrow_Close'] = np.where(df['Tomorrow_Close'] >= df['Close'], 1.0, 0.0)
        self._data = df
    


    def run():
        print("run implemented")
    
    def post_results():
        x = 1