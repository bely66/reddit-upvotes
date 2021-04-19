from pickle import NONE
import torch
from torch import nn
from torch._C import dtype
import warnings
warnings.filterwarnings('ignore')
class MLPRegressor(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, input_shape):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(input_shape, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)



class TrainRegressor():
    def __init__(self, mlp=None, epochs=5, train=True) -> None:
        # Define the loss function and optimizer
        self.model = None
        self.epochs = epochs
        if mlp is not None:
            self.model = mlp
        else:
            self.model = self.load_model(train=train)

        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)


        
        
    def fit(self, i, x, y):
        self.model.train()
        for epoch in range(0, self.epochs): # 5 epochs at maximum            
            # Set current loss value
            current_loss = 0.0
            
            # Iterate over the DataLoader for training data
            
            # Get inputs
            inputs, targets = torch.from_numpy(x).float(), torch.from_numpy(y).float()
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Perform forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            loss = self.loss_function(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            self.optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss/10))
            current_loss = 0.0

        # Process is complete.
        print('Training process has finished.')
        self.save_model()

    def save_model(self, PATH="regression_model.pt"):
        self.model.eval()
        torch.save(self.model, PATH)

    def load_model(self, PATH="regression_model.pt", train=False):
        model = torch.load(PATH)
        if train:
            model.train()
        return model

    def predict(self, inputs):
        inputs = torch.from_numpy(inputs).float()
        outputs = self.model(inputs).detach().numpy()

        return outputs

