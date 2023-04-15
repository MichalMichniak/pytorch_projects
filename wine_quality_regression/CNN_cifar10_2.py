from typing import Tuple,List
from tqdm import tqdm
import torch
from torch.utils.data import Dataset,DataLoader
class CNN_cifar10_2:
    def __init__(self) -> None:
        self.conv1 = torch.nn.Conv2d(3,9,5) # 18 chanels 28x28
        self.poling1 = torch.nn.AvgPool2d(2,2) # 18 chanels 14x14
        self.conv2 = torch.nn.Conv2d(9,18,5) # 28 channels 10x10
        self.poling2 = torch.nn.AvgPool2d(2,2) # 28 channels 5x5
        self.conv3 = torch.nn.Conv2d(18,32,3) # 32 channels 3x3
        self.flatten1 = torch.nn.Flatten(-3,2) # 288
        self.FC1 = torch.nn.Linear(288, 20)
        self.FC2 = torch.nn.Linear(20,12)
        self.out = torch.nn.Linear(12,10)
        self.dropout1 = torch.nn.Dropout(p=0.1)
        self.dropout2 = torch.nn.Dropout(p=0.05)
        self.model = torch.nn.Sequential(
            self.conv1,
            self.poling1,
            self.conv2,
            self.poling2,
            self.conv3,
            self.flatten1,
            self.FC1,
            torch.nn.ReLU(),
            self.dropout1,
            self.FC2,
            torch.nn.ReLU(),
            self.dropout2,
            self.out,
            torch.nn.Softmax(dim = -1)
        )

    def forward_pass(self, x : torch.Tensor):
        return self.model(x.T)
    
    def train(self,nr_epoch, dataloader : DataLoader, x_test, y_test, learning_rate = 1e-4)->Tuple[List[float],List[float],List[float],List[float]]:
        """
        param:
            nr_epoch : Uint - number of epochs
            dataloader : DataLoader - dataloader for training process
            x_test : np.ndarray - test data
            y_test : np.ndarray - class of test data (evaluation)
            learning_rate : float - learning rate for backpropagationsteps
        return:
            train_loss : List[float] - training loss
            test_loss : List[float] - test loss
            train_accuracy : List[float] - training accuracy
            test_accuracy : List[float] - test accuracy
        """
        test_accuracy = []
        train_accuracy = []
        test_loss = []
        train_loss = []

        mean_loss = 0
        accuracy = 0
        optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.1, lr = learning_rate)
        optimizer.zero_grad()
        loss_func = torch.nn.CrossEntropyLoss()
        for epoch in range(nr_epoch):
            mean_loss = 0
            accuracy = 0
            for (n,(x_batch,y_batch)),tqdm_progress in zip(enumerate(dataloader),tqdm(range(len(dataloader)))):
                
                for x,y in zip(x_batch,y_batch):
                    y_pred = self.forward_pass(x)
                    loss = loss_func(y_pred,y)
                    with torch.no_grad():
                        mean_loss += loss
                        if torch.argmax(y_pred) == torch.argmax(y):
                            accuracy += 1
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            mean_loss = mean_loss/len(dataloader)
            train_loss.append(mean_loss)
            accuracy = accuracy/len(dataloader)
            train_accuracy.append(accuracy)
            mean_loss = 0
            accuracy = 0
            with torch.no_grad():
                for x,y in zip(x_test, y_test):
                    y_pred = self.forward_pass(x)
                    loss = loss_func(y_pred,y)
                    mean_loss += loss
                    if torch.argmax(y_pred) == torch.argmax(y):
                        accuracy += 1
                mean_loss = mean_loss/len(dataloader)
                test_loss.append(mean_loss)
                accuracy = accuracy/len(dataloader)
                test_accuracy.append(accuracy)
            print(f"EPOCH: {epoch+1}, TEST_LOSS{mean_loss}, TEST_ACCURACY{accuracy}")
        return train_loss,test_loss,train_accuracy,test_accuracy