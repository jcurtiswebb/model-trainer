import unittest
from Spx1dLstmTrainer import Spx1dLstmModelTrainer

class Test_ModelTrainer(unittest.TestCase):
    def test_constructor(self):
        trainer = Spx1dLstmModelTrainer()
        self.assertIsNotNone(trainer)


    def test_model_create(self):
        trainer = Spx1dLstmModelTrainer()
        kwargs = {'input_size':10, 'hidden_size':4, 'num_layers':2, 'fc_neurons':8}
        trainer.update_model(**kwargs)
        self.assertEqual(trainer.model.input_size, 10)
        self.assertEqual(trainer.model.hidden_size, 4)
        self.assertEqual(trainer.model.num_layers, 2)

    def test_data_update(self):
        trainer = Spx1dLstmModelTrainer()
        trainer.update_data()
        self.assertGreaterEqual(trainer.data.shape[0], 500)

    def test_post_results(self):
        # Make a throw-away experiment in MLFlow
        
        # Make an Spx1dLstmModelTrainer

        # Make dummy results (don't actually use Pytorch)

        # Post the dummy results to MLflow

        # Read the results back from MLFlow
        # unit test assert read-results == posted dummy results

        # Delete the throw-away experiment from MLFlow
        self.assertEqual(True, False)

if __name__ == '__main__':
    unittest.main()