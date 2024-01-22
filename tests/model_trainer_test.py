import unittest
from Spx1dLstmTrainer import Spx1dLstmModelTrainer
import mlflow

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

    def test_new_experiment(self):
        trainer = Spx1dLstmModelTrainer()
        
        # Make a throw-away experiment in MLFlow
        test_date = '01_17_2024'
        trainer._last_trainable_trade_date = test_date

        trainer.update_experiment()
  
        self.assertTrue(test_date in trainer.experiment.name)

        # TODO Need to wrap this delete experiment in a finally
        mlflow.delete_experiment(trainer.experiment.experiment_id)

    def test_post_results(self):
        trainer = Spx1dLstmModelTrainer()
        
        # Make a throw-away experiment in MLFlow
        test_date = 'aa_01-17-2024'


        trainer.update_experiment(test_date)

        # Make dummy results (don't actually use Pytorch)

        # Post the dummy results to MLflow

        # Read the results back from MLFlow
        # unit test assert read-results == posted dummy results

        # Delete the throw-away experiment from MLFlow
        self.assertEqual(True, False)

    def test_throwaway_do_we_populate_mlflow_with_1_16(self):
        trainer = Spx1dLstmModelTrainer()
        trainer.update_data_and_experiment()

        self.assertTrue("01_16_2024" in trainer.experiment.name)

if __name__ == '__main__':
    unittest.main()