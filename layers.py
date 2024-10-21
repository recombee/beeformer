import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import sentence_transformers
import torch

from keras.layers import TorchModuleWrapper


# basic elsa model as a keras layer (usebale at other keras models)
class LayerELSA(keras.layers.Layer):
    def __init__(self, n_dims, n_items, device):
        super(LayerELSA, self).__init__()
        self.device = device
        self.A = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty([n_dims, n_items])))

    def parameters(self, recurse=True):
        return [self.A]

    def track_module_parameters(self):
        for param in self.parameters():
            variable = keras.Variable(initializer=param, trainable=param.requires_grad)
            variable._value = param
            self._track_variable(variable)
        self.built = True

    def build(self):
        self.to(self.device)
        sample_input = torch.ones([self.A.shape[0]]).to(self.device)
        _ = self.call(sample_input)
        self.track_module_parameters()

    def call(self, x):
        A = torch.nn.functional.normalize(self.A, dim=-1)
        xA = torch.matmul(x, A)
        xAAT = torch.matmul(xA, A.T)
        return keras.activations.relu(xAAT - x)


# keras wrapper around sentence transformers objet
class LayerSBERT(keras.layers.Layer):
    def __init__(self, model, device):
        super(LayerSBERT, self).__init__()
        self.device = device
        self.sbert = TorchModuleWrapper(model.to(device))
        self.tokenize_ = self.sb().tokenize
        self.build()

    def sb(self):
        for module in self.sbert.modules():
            if isinstance(module, sentence_transformers.SentenceTransformer):
                return module

    def parameters(self, recurse=True):
        return self.sbert.parameters()

    def track_module_parameters(self):
        for param in self.parameters():
            variable = keras.Variable(initializer=param, trainable=param.requires_grad)
            variable._value = param
            self._track_variable(variable)
        self.built = True

    def tokenize(self, inp):
        # move tokenized tensors to device and return tokenized sentences
        return {k: v.to(self.device) for k, v in self.tokenize_(inp).items()}

    def build(self):
        self.to(self.device)
        sample_input = ["text", "text2"]
        inp = self.tokenize(sample_input)
        _ = self.call(inp)
        self.track_module_parameters()

    def call(self, x):
        # just call sentence transformer model
        return self.sbert.forward(x)["sentence_embedding"]