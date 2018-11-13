import torch
import logging
from torch import nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable

__all__ = ["Trainer"]


class Trainer:
    """
    Trainer class, provides network training regardless of the architecture.
    """


    def __init__(self, model, train, test, logdir="logs"):
        assert (model is not None), "Empty model"
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters())
        self.loss_function = nn.CrossEntropyLoss()
        self.train = train
        self.test = test
        self.logger = SummaryWriter(logdir)

    def train(self, optimizer=None, loss_function=None, epochs=50):
        """
        Train void function.
        :param optimizer: Optimizer to minimize loss function. Adam by default
        :param loss_function: Loss function to train on. CrossEntropy by default
        :param epochs: Number of epochs to run
        """
        if self.optimizer is not None:
            self.optimizer = optimizer
        if self.loss_function is not None:
            self.loss_function = loss_function
        logging.info("Started training")
        for i in range(epochs):
            (train_acc, train_loss), (test_acc, test_loss) = self.train_epoch()
            logging.debug("train_acc: {}, train_loss: {}, test_acc: {}, test_loss: {}".format(
                train_acc, train_loss, test_acc, test_loss))
            self.logger.add_scalars('accuracy', {'train': train_acc}, i)
            self.logger.add_scalars('accuracy', {'test': test_acc}, i)
            self.logger.add_scalars('loss', {'train': train_loss}, i)
            self.logger.add_scalars('loss', {'test': test_loss}, i)

    def train_epoch(self):
        """
        Function to run one epoch.
        :return: two tuples for scores of train and test, each tuple contains acuracy and loss
        """
        for data, labels in self.train:
            self.optimizer.zero_grad()
            prediction = self.model.forward(data)
            loss = self.loss_function.forward(prediction, labels)
            loss.backward()
            self.optimizer.step()
        return self.count_scores(self.train), self.count_scores(self.test)

    def count_scores(self, dataset):
        """
        Function to count accuracy and loss on given dataset.
        :param dataset: Dataloader that contains data to count scores on
        :return: accuracy and loss
        """
        n = 0
        avg_loss = 0.
        avg_accuracy = 0.
        for data, labels in dataset:
            n += 1
            predicted = self.predict(data)
            avg_accuracy += (predicted == labels).sum().item() / len(data)
            loss = self.loss_function.forward(predicted, labels)
            avg_loss += loss.item()
        avg_loss /= n
        avg_accuracy /= n
        return avg_accuracy, avg_loss

    def predict(self, data):
        """
        Runs model on given data
        :param data: data to predict
        :return: predicted labels
        """
        data = Variable(data, requires_grad=False)
        data_out = self.model(data)
        return data_out.data.numpy().argmax(axis=1)
