import mlflow
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
import torch
from joblib import dump, load
import pickle

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

# TODO : Consider making the training loop part of the ModelTrainer class, or maybe an inherited class from ModelTrainer like ModelTrainerBinCls
def training_loop_bin_class(n_epochs, model, optimiser, loss_fn, X_train, y_train,
                  X_test = None, y_test = None, early_stopper = None):
    train_metrics = []
    for epoch in range(n_epochs):
        model.train()
        outputs = model.forward(X_train) # forward pass
        optimiser.zero_grad() # calculate the gradient, manually setting to 0
        # obtain the loss function
        loss = loss_fn(outputs, y_train)
        loss.backward() # calculates the loss of the loss function
        
        # obtain the accuracy
        acc = (outputs.round() == y_train).float().mean()
        
        # consider gradient clipping / normalization
        # torch.nn.utils.clip_grad_norm(lstm.parameters(), max_norm=1)
        
        optimiser.step() # improve from loss, i.e backprop
                
        # test loss
        if X_test is not None and y_test is not None:
            model.eval()
            test_preds = model(X_test)
            test_loss = loss_fn(test_preds, y_test)
            test_acc = (test_preds.round() == y_test).float().mean()
            train_metrics.append({
                'epoch':epoch+1, 
                'train_loss':loss.item(),
                'train_acc':acc.item(), 
                'test_loss':test_loss.item(), 
                'test_acc':test_acc.item()
            })
            
            if early_stopper is not None:
                if early_stopper.early_stop(test_loss.item()):
                    return epoch+1, train_metrics
        else:
            train_metrics.append({
                'epoch':epoch+1, 
                'train_loss':loss.item(),
                'train_acc':acc.item()
            })
    
    return n_epochs, train_metrics

def add_model_to_artifacts(model, model_name, artifacts):
    # Save the model and make it an artifact
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_filename = f"temp/last_feature_date_{model_name}.pt"
    model_scripted.save(model_filename) # Save
    artifacts.append(model_filename)

def log_result_mlflow(params, metrics, df_metrics, lstm_acc, artifacts, run):
    mlflow_client = MlflowClient()
    # Log the parameters used for the model fit
    mlflow.log_params(params)

    # Log the error metrics that were calculated during validation
    mlflow.log_metrics(metrics)

    epoch_metrics = []
    cols = df_metrics.columns
    for epoch in df_metrics.index:
        current_epoch_metrics = {col:df_metrics.loc[epoch,col] for col in df_metrics.columns}
        for k,v in current_epoch_metrics.items():
            metric = Metric(key=k,
                            value=v,
                            timestamp=0,
                            step=epoch)
            epoch_metrics.append(metric)
#             mlflow.log_metrics(current_epoch_metrics, step=epoch)
    mlflow_client.log_batch(run_id = run.info.run_id, metrics=epoch_metrics)

    # Log an instance of the trained model for later use
    for key, local_path in artifacts.items():
        mlflow.log_artifact(local_path)

    mlflow.log_input(dataset, context="training")

# create class
class PyFuncPyTorchLSTM(mlflow.pyfunc.PythonModel):          
    def predict(self, context, X, y=None):
        print('predict')
        sl = self.parameters['seq_len']
        print('loaded sequence length')
        print(X)
        X = X[0].iloc[-sl:]
        print(f'loaded data : {X.shape}')
        # We need to rescale the new data using standard scaler
        
        X_ss = self.ss.transform(X)
        print('Transformed X')
        # We need to used scaled data and make prediction

        X_tensors = torch.unsqueeze(torch.Tensor(X_ss),0)
        print('Created Tensor')

        self.model.device = torch.device("cpu") 
        self.model.to(self.model.device)
        self.model.eval()
        print('Turned on model eval')
        pred = self.model(X_tensors)
        print('obtained prediction')
        output = pred.detach().cpu().numpy()
        print('detatched prediction')
        # 1 is a gain prediction
        # 0 is a loss prediction
        return output
    
    def load_context(self, context):
        # Standard Scaler artifact
        self.ss = load(context.artifacts['ss'])
        # Pytorch Model artifact
        self.model = torch.jit.load(context.artifacts['mdl'])

        # Model parameters artifact
        with open(context.artifacts['params'], 'rb') as f:
            self.parameters = pickle.load(f)

        print('loaded contexts')