from abc import ABC, abstractmethod

class ModelTrainer(ABC):
    def __init__(self):
        self.schedule = {'M':('16:15:00', 23.5), 'T':('16:15:00', 23.5), 'W':('16:15:00', 23.5), 'Th':('16:15:00', 23.5), 'F':('16:15:00', 23.5)}
        self.pred_type = 'bc'
        self.model = None
        self._data = None
        self._hyperparams = None
         
    @property
    def data(self):
        return self._data
    
    @data.setter
    def update_data(self, **kwargs):
        self._data = self._data

    @property
    def hyperparams(self):
        return self._hyperparams
    
    @hyperparams.setter
    def update_hyperparams(self, **kwargs):
        self._hyperparams = self._hyperparams
    
    @abstractmethod
    def run():
        pass

    




