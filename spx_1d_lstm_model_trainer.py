from model_trainer import ModelTrainer
from model_store import LSTM1
from utils.y_finance_utils import *
from utils.model_trainer_utils import *
from datetime import time
import time as t
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')


class Spx1dLstmModelTrainer(ModelTrainer):
    def update_model(self, **kwargs):
        input_size = self.data.shape[1]-1
        hidden_size = kwargs['hidden_size']
        num_layers = kwargs['num_layers']
        fc_neurons = kwargs['fc_neurons']
        self._model_1 = LSTM1(input_size, hidden_size, num_layers, fc_neurons, self._device_1)
        self._model_2 = LSTM1(input_size, hidden_size, num_layers, fc_neurons, self._device_2)

    def update_data(self, **kwargs):
        market_close_time = time(14,15)
        current_time = datetime.now().time()
        if current_time > market_close_time or self._last_trainable_trade_date is None:
            df = get_latest_data()

            if datetime.now().strftime('%m_%d_%Y') == df.iloc[-1].name.strftime('%m_%d_%Y') and current_time < market_close_time:
                df = df.iloc[:-1]

            col = df.pop('Close')
            df.insert(df.shape[1], 'Close', col)
            df['Tomorrow_Close'] = df['Close'].shift(-1)
            df.dropna(inplace=True)
            # Turn tomorrow close into a binary classification
            df['Tomorrow_Close'] = np.where(df['Tomorrow_Close'] >= df['Close'], 1.0, 0.0)


            self._last_trainable_trade_date = df.iloc[-1].name.strftime('%m_%d_%Y')
            self._data = df

    
    def determine_epochs(self, device):
        start_idx, end_idx = TrainerUtils.get_start_end_idx(self.data)
        cv_epochs = []

        for i in range(start_idx, end_idx, 1):
            df_cv = TrainerUtils.make_cross_val_df(self.data, i, self._hyperparams['sliding_window'])
            input_size = df_cv.shape[1]-1
            
            ss = StandardScaler()
            X_train_tensors, y_train_tensors, X_test_tensors, y_test_tensors = self.make_tensors(ss, df_cv, self._hyperparams['seq_len'], device)
            
            train_val_split = int(X_train_tensors.shape[0]-1)

            model = LSTM1(input_size, self._hyperparams['hidden_size'], self._hyperparams['num_layers'], self._hyperparams["fc_neurons"], device).to(device)
            loss_fn = torch.nn.BCELoss()    # mean-squared error for regression
            # Foreach set to false because of PyTorch bug.
            optimiser = torch.optim.Adam(model.parameters(), lr=self._hyperparams['learning_rate'])
            early_stopper = EarlyStopper()


            n_completed_epochs, train_metric = TrainerUtils.training_loop_bin_class(
                n_epochs= self._hyperparams['n_epochs'],
                model=model,
                optimiser=optimiser,
                loss_fn=loss_fn,
                X_train=X_train_tensors[:train_val_split],
                y_train=y_train_tensors[:train_val_split], 
                X_test=X_train_tensors[train_val_split:], 
                y_test=y_train_tensors[train_val_split:],
                early_stopper=early_stopper
            )
            cv_epochs.append(n_completed_epochs)
            
        return int(sum(cv_epochs)/len(cv_epochs))
    

    def run(self):
        device = self._device_1
        start_idx, end_idx = TrainerUtils.get_start_end_idx(self.data)

        df_plot = pd.DataFrame()
        train_metrics = []
        artifacts = {}
        start_time = t.time()

        # Initiate the MLflow run context
        with mlflow.start_run(run_name = str(self.hyperparams)) as run:
            n_epochs = self.determine_epochs(device)
            
            for i in range(start_idx, end_idx, 1):
                # print(f"Processing : {df_with_final_row.index[i]}")

                df_cv = TrainerUtils.make_cross_val_df(self.data, i, self.hyperparams['sliding_window'])

                ss = StandardScaler()
                X_train_tensors, y_train_tensors, X_test_tensors, y_test_tensors = self.make_tensors(ss, df_cv, self.hyperparams['seq_len'], device)
                
                # TODO: Might need to recreate the model each loop
                self.update_model(**self.hyperparams)

                model = self._model_1.to(self._model_1.device)
                loss_fn = torch.nn.BCELoss()    # mean-squared error for regression
                # Foreach set to false because of PyTorch bug.
                optimiser = torch.optim.Adam(model.parameters(), lr=self.hyperparams['learning_rate'])


                n_epochs, train_metric = TrainerUtils.training_loop_bin_class(
                    n_epochs=n_epochs,
                    model=model,
                    optimiser=optimiser,
                    loss_fn=loss_fn,
                    X_train=X_train_tensors,
                    y_train=y_train_tensors, 
                    X_test=X_test_tensors, 
                    y_test=y_test_tensors
                )

                train_metrics = [*train_metrics, *train_metric]
                
                if i == end_idx - 1:
                    # We serialize our final standard scaler for inference
                    os.makedirs(self.CONFIG['artifact_directory']+f'/{run.info.run_id}/', exist_ok=True)
                    dump(ss, self.CONFIG['artifact_directory']+f'/{run.info.run_id}/standard_scalar.pkl')
                    artifacts['ss']=self.CONFIG['artifact_directory']+f'/{run.info.run_id}/standard_scalar.pkl'
                    
                    # Save the model and make it an artifact
                    model_scripted = torch.jit.script(model) # Export to TorchScript
                    model_filename = self.CONFIG['artifact_directory']+f'/{run.info.run_id}/model.pt'
                    model_scripted.save(model_filename) # Save
                    artifacts['mdl']=model_filename

                    # Save params as a dictionary in a file
                    param_dict_filename = self.CONFIG['artifact_directory']+f'/{run.info.run_id}/param_dict.pkl'
                    with open(param_dict_filename, 'wb') as f:
                        pickle.dump(self.hyperparams, f)
                    artifacts['params']=param_dict_filename

                data_predict = model(X_test_tensors).cpu().data.numpy() # numpy conversion
                dataY_actual = np.array(df_cv.iloc[:,-1:].values[-1:])
                df_plot = pd.concat([df_plot, pd.DataFrame([
                    {'Actual Data':dataY_actual[0][0], 'Predicted Data':data_predict[0][0],'Date':self.data.index[i]}
                ])])

            
            end_time = t.time()

            df_metrics = pd.DataFrame(train_metrics)
            df_metrics = df_metrics.groupby('epoch').mean()

            df_plot['Predicted Data'] = df_plot['Predicted Data'].round()
            lstm_acc = (df_plot['Actual Data']==df_plot['Predicted Data']).mean()
            # Assemble the metrics we're going to write into a collection
            metrics = {"Accuracy": lstm_acc, "Elapsed" : round(end_time - start_time,1)}
            print(f"Overall accuracy of {str(self.hyperparams)} : {lstm_acc}")
            
            self.hyperparams['model_path'] = self.CONFIG['artifact_directory']+f'/{run.info.run_id}/model'
            mlflow.pyfunc.save_model(path=self.hyperparams['model_path'], python_model=PyFuncPyTorchLSTM(), artifacts=artifacts)

            TrainerUtils.log_result_mlflow(self.hyperparams, metrics, df_metrics, artifacts, run)
    