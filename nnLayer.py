from torch import nn as nn
from torch.nn import functional as F
import torch,time,os
import numpy as np

class TextEmbedding(nn.Module):
    def __init__(self, embeding, freeze=False, dropout=0.2, name='textEmbedding'):
        super(TextEmbedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embeding, dtype=torch.float32), freeze=freeze)
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x):
        return self.dropout(self.embedding(x))

class TextCNN(nn.Module):
    def __init__(self, featureSize, filterSize, contextSizeList, reduction='pool', actFunc=nn.ReLU(), name='textCNN'):
        super(TextCNN, self).__init__()
        moduleList = []
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=featureSize, out_channels=filterSize, kernel_size=contextSizeList[i], padding=contextSizeList[i]//2),
                    #nn.BatchNorm1d(filterSize),
                    actFunc
                    )
                )
        self.conv1dList = nn.ModuleList(moduleList)
        self.reduction = reduction
        self.name = name
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1,2) # => batchSize × feaSize × seqLen
        x = [conv(x).squeeze(dim=2) for conv in self.conv1dList] # => scaleNum * (batchSize × filterSize × seqLen)
        if self.reduction=='pool':
            x = [F.adaptive_max_pool1d(i, 1).squeeze(dim=2) for i in x]
            return torch.cat(x, dim=1) # => batchSize × scaleNum*filterSize
        if self.reduction=='GCN':
            x = [F.adaptive_max_pool1d(i, 1) for i in x]
            return torch.cat(x, dim=2) # => batchSize × filterSize × scaleNum
        elif self.reduction=='None':
            return [i.transpose(1,2) for i in x] # => scaleNum * (batchSize × seqLen × filterSize)
         
class TextGRU(nn.Module):
    def __init__(self, feaSize, hiddenSize, num_layers=1, dropout=0.0, bidirectional=False, name='textBiGRU'):
        super(TextGRU, self).__init__()
        self.name = name
        self.biGRU = nn.GRU(feaSize, hiddenSize, bidirectional=bidirectional, batch_first=True, num_layers=num_layers, dropout=dropout)
    def forward(self, x, xlen=None):
        # x: batchSizeh × seqLen × feaSize
        if xlen is not None:
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)

            x = nn.utils.rnn.pack_padded_sequence(x[indices], xlen, batch_first=True)
        output, hn = self.biGRU(x) # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize
        if xlen is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output[desortedIndices]
        return output # output: batchSize × seqLen × hiddenSize*2

class SimpleAttention(nn.Module):
    def __init__(self, inSize, actFunc=nn.ReLU(), name='SimpleAttention', transpose=False):
        super(SimpleAttention, self).__init__()
        self.name = name
        self.W = nn.Parameter(torch.randn(size=(inSize,1), dtype=torch.float32))
        self.actFunc = actFunc
        self.transpose = transpose
    def forward(self, input):
        if self.transpose:
            input = input.transpose(1,2)
        # input: batchSize × seqLen × inSize
        #H = self.actFunc(input) # => batchSize × seqLen × inSize
        alpha = F.softmax(torch.matmul(input,self.W), dim=1) # => batchSize × seqLen × 1
        return self.actFunc( torch.matmul(input.transpose(1,2), alpha).squeeze(2) ) # => batchSize × inSize

class SimpleAttention2(nn.Module):
    def __init__(self, inSize, outSize, actFunc=nn.Tanh, name='SimpleAttention2', transpose=False):
        super(SimpleAttention2, self).__init__()
        self.name = name
        self.W = nn.Parameter(torch.randn(size=(outSize,1), dtype=torch.float32))
        self.attnWeight = nn.Sequential(
            nn.Linear(inSize, outSize),
            actFunc()
            )
        self.transpose = transpose
    def forward(self, input):
        if self.transpose:
            input = input.transpose(1,2)
        # input: batchSize × seqLen × inSize
        H = self.attnWeight(input) # => batchSize × seqLen × outSize
        alpha = F.softmax(torch.matmul(H,self.W), dim=1) # => batchSize × seqLen × 1
        return torch.matmul(input.transpose(1,2), alpha).squeeze(2) # => batchSize × inSize

class KnowledgeAttention(nn.Module):
    def __init__(self, noteFeaSize, titleFeaSize, name='knowledgeAttention'):
        super(KnowledgeAttention, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(noteFeaSize, titleFeaSize),
            nn.Tanh()
            )
        self.labWeight = None
        self.name = name
    def forward(self, noteConved, titleEncoded):
        # noteConved: batchSize × noteFeaSize; titleEncoded: titleNum × titleFeaSize
        x = self.linear(noteConved) # => batchSize × titleFeaSize
        attnWeight = F.softmax(torch.matmul(x, titleEncoded.transpose(0,1)), dim=1) # => batchSize × titleNum
        self.labWeight = attnWeight.detach().cpu().numpy()
        return torch.matmul(attnWeight, titleEncoded) # => batchSize × titleFeaSize

class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenSizeList=[], dropout=0.1, name='MLP', actFunc=nn.ReLU):
        super(MLP, self).__init__()
        self.name = name
        layers = nn.Sequential()
        for i,os in enumerate(hiddenSizeList):
            layers.add_module(str(i*2), nn.Linear(inSize, os))
            layers.add_module(str(i*2+1), actFunc())
            inSize = os
        self.hiddenLayers = layers
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(inSize, outSize)
    def forward(self, x):
        x = self.hiddenLayers(x)
        return self.out(self.dropout(x))
