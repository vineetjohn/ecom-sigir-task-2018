import torch

from src_neural.config.model_config import mconf


class NeuralClassifier(torch.nn.Module):

    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.dropout = torch.nn.Dropout(mconf.dropout)
        self.fc1 = torch.nn.Linear(vocab_size, mconf.layer_1_size)
        self.fc2 = torch.nn.Linear(mconf.layer_1_size, mconf.layer_2_size)
        self.fc3 = torch.nn.Linear(mconf.layer_2_size, num_classes)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        x = torch.autograd.Variable(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        logits = self.softmax(x)

        return logits
