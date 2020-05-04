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
         

class TextICDAttentionalCNN(nn.Module):
    def __init__(self, featureSize, contextSizeList, filterSize, classNum, name='textCNN'):
        super(TextICDAttentionalCNN, self).__init__()
        moduleList = []
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=featureSize, out_channels=filterSize, kernel_size=contextSizeList[i]),
                    nn.ReLU(),
                    ICDAttention(filterSize, classNum=classNum, transpose=True)
                    )
                )
        self.conv1dList = nn.ModuleList(moduleList)
        self.linear = nn.Linear(len(contextSizeList)*filterSize, classNum)
        self.classNum = classNum
        self.name = name
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1,2) # => batchSize × feaSize × seqLen
        x = [conv(x).squeeze(dim=1) for conv in self.conv1dList] # => scaleNum * (batchSize × filterSize × classNum)
        x = torch.cat(x, dim=1).transpose(1,2) # => batchSize × classNum × scaleNum*filterSize
        return self.linear.weight.mul(x).sum(dim=2) + self.linear.bias # => batch × classNum

class TextAYNICNN(nn.Module):
    def __init__(self, featureSize, contextSizeList=[1,3,5], name='textAYNICNN'):
        super(TextAYNICNN, self).__init__()
        moduleList = []
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=featureSize, out_channels=featureSize, kernel_size=contextSizeList[i], padding=contextSizeList[i]//2),
                    )
                )
        self.feaConv1dList = nn.ModuleList(moduleList)
        self.attnConv1d = nn.Sequential(
                            nn.Conv1d(in_channels=featureSize*len(contextSizeList), out_channels=1, kernel_size=5, padding=2),
                            nn.Softmax(dim=2)
                          )
        self.name = name
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1,2) # => batchSize × feaSize × seqLen
        x = torch.cat([conv(x)+x for conv in self.feaConv1dList],dim=1) # => batchSize × filterSize*contextNum × seqLen
        x = torch.tanh(x) # => batchSize × (feaSize*contextNum) × seqLen
        alpha = self.attnConv1d(x).transpose(1,2) # => batchSize × seqLen × 1
        return torch.matmul(x, alpha).squeeze(dim=2) # => batchSize × (filterSize*contextNum)

class TextAttention(nn.Module):
    def __init__(self, method, name='textAttention'):
        super(TextAttention, self).__init__()
        self.attn = LuongAttention(method)
        self.name = name
    def forward(self, sequence, reference):
        # sequence: batchSize × seqLen × feaSize; reference: batchSize × classNum × feaSize
        alpha = self.attn(reference, sequence) # => batchSize × classNum × seqLen
        return torch.matmul(alpha, sequence) # => batchSize × classNum × feaSize

class LuongAttention(nn.Module):
    def __init__(self, method):
        super(LuongAttention, self).__init__()
        self.method = method
    def dot_score(self, hidden, encoderOutput):
        # hidden: batchSize × classNum × hiddenSize; encoderOutput: batchSize × seq_len × hiddenSize
        return torch.matmul(encoderOutput, hidden.transpose(-1,-2)) # => batchSize × seq_len × classNum
    def forward(self, hidden, encoderOutput):
        attentionScore = self.dot_score(hidden, encoderOutput).transpose(-1,-2)
        # attentionScore: batchSize × classNum × seq_len
        return F.softmax(attentionScore, dim=-1) # => batchSize × classNum × seq_len

class Transformer(nn.Module):
    def __init__(self, featureSize, dk, multiNum, seqMaxLen, dropout=0.1):
        super(Transformer, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WK = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WV = nn.ModuleList([nn.Linear(featureSize, self.dk) for i in range(multiNum)])
        self.WO = nn.Linear(self.dk*multiNum, featureSize)
        self.layerNorm1 = nn.LayerNorm([seqMaxLen, featureSize])
        self.layerNorm2 = nn.LayerNorm([seqMaxLen, featureSize])
        self.Wffn = nn.Sequential(
                        nn.Linear(featureSize, featureSize*4), 
                        nn.ReLU(),
                        nn.Linear(featureSize*4, featureSize)
                    )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        queries = [self.WQ[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        keys = [self.WK[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        values = [self.WQ[i](x) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        score = [torch.bmm(queries[i], keys[i].transpose(1,2))/np.sqrt(self.dk) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × seqLen)
        z = [self.dropout(torch.bmm(F.softmax(score[i], dim=2), values[i])) for i in range(self.multiNum)] # => multiNum*(batchSize × seqLen × dk)
        z = self.WO(torch.cat(z, dim=2)) # => batch × seqLen × feaSize
        z = self.layerNorm1(x + z) # => batchSize × seqLen × feaSize
        ffnx = self.Wffn(z) # => batchSize × seqLen × feaSize
        return self.layerNorm2(z + ffnx) # => batchSize × seqLen × feaSize

class TextTransformer(nn.Module):
    def __init__(self, layersNum, featureSize, dk, multiNum, seqMaxLen, dropout=0.1, name='textTransformer'):
        super(TextTransformer, self).__init__()
        posEmb = [[np.sin(pos/10000**(2*i/featureSize)) if i%2==0 else np.cos(pos/10000**(2*i/featureSize)) for i in range(featureSize)] for pos in range(seqMaxLen)]
        self.posEmb = nn.Parameter(torch.tensor(posEmb, dtype=torch.float32), requires_grad=False) # seqLen × feaSize
        self.transformerLayers = nn.Sequential(
                                        OrderedDict(
                                            [('transformer%d'%i, Transformer(featureSize, dk, multiNum, seqMaxLen, dropout)) for i in range(layersNum)]
                                        )
                                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = self.dropout(x+self.posEmb) # => batchSize × seqLen × feaSize
        return self.transformerLayers(x) # => batchSize × seqLen × feaSize

class Siamese(nn.Module):
    def __init__(self, featureSize, name='siamese'):
        super(Siamese, self).__init__()
        self.out = nn.Linear(featureSize, 1)
                    #nn.Sequential(
                    #    nn.Linear(featureSize, 1),
                    #    nn.Sigmoid()
                    #)
        self.name = name
    def forward(self, noteFea, labFea):
        # noteFea: batchSize × feaSize; labFea: labNum × feaSize
        dis =  noteFea.unsqueeze(1) - labFea.unsqueeze(0) # => batchSize × labNum × feaSize
        return self.out(dis).squeeze(2) # => batchSize × labNum
        #dis = -self.out(dis).squeeze(2) # => batchSize × labNum
        #return torch.exp(dis) # => batchSize × labNum

class TextLSTM(nn.Module):
    def __init__(self, feaSize, hiddenSize, bidirectional=False, name='textBiLSTM'):
        super(TextLSTM, self).__init__()
        self.name = name
        self.biLSTM = nn.LSTM(feaSize, hiddenSize, bidirectional=bidirectional, batch_first=True)
    def forward(self, x, xlen=None):
        # x: batchSizeh × seqLen × feaSize
        if xlen is not None:
            xlen, indices = torch.sort(xlen, descending=True)
            _, desortedIndices = torch.sort(indices, descending=False)

            x = nn.utils.rnn.pack_padded_sequence(x[indices], xlen, batch_first=True)
        output, hn = self.biLSTM(x) # output: batchSize × seqLen × hiddenSize*2; hn: numLayers*2 × batchSize × hiddenSize
        if xlen is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            return output[desortedIndices]
        return output # output: batchSize × seqLen × hiddenSize*2

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

class ICDAttention(nn.Module):
    def __init__(self, inSize, classNum, transpose=False, name='ICDAttention'):
        super(ICDAttention, self).__init__()
        self.transpose = transpose
        self.U = nn.Linear(inSize, classNum)
    def forward(self, X):
        if self.transpose:
            X = X.transpose(1,2)
        # X: batchSize × seqLen × inSize
        alpha = F.softmax(self.U(X), dim=1) # => batchSize × seqLen × classNum
        X = torch.matmul(X.transpose(1,2), alpha) # => batchSize × inSize × classNum
        return X

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

class GCN(nn.Module):
    def __init__(self, nodeNum, L, feaSize, hiddenSizeList=[], embDropout=0.1, name='GCN'):
        super(GCN, self).__init__()
        self.name = name
        self.embDropout = nn.Dropout(p=embDropout)
        self.L = nn.Parameter(torch.tensor(L, dtype=torch.float32), requires_grad=False) # noteNum × nodeNum
        inSize,hiddens = feaSize,[]
        for i,os in enumerate(hiddenSizeList):
            hiddens.append(nn.Linear(inSize, os))
            inSize = os
        self.hiddens = nn.ModuleList(hiddens)
    def forward(self, x):
        # x: batchSize × nodeNum × feaSize;
        out = [] 
        out.append(self.embDropout(x))
        for h in self.hiddens:
            out.append(F.relu(h(torch.matmul(out[-1].transpose(1,2), self.L).transpose(1,2))))
        return out # => layerNum * (batchSize × nodeNum × outSize)

class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gama=2, weight=None, logit=True):
        super(FocalCrossEntropyLoss, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32), requires_grad=False) if weight is not None else weight
        self.gama = gama
        self.logit = logit
    def forward(self, Y_pre, Y):
        if self.logit:
            Y_pre = F.softmax(Y_pre, dim=1)
        P = Y_pre[list(range(len(Y))), Y]
        if self.weight is not None:
            w = self.weight[Y]
        else:
            w = torch.tensor([1.0 for i in range(len(Y))], device=self.weight.device())
        w = (w/w.sum()).reshape(-1)
        return (-w*((1-P)**self.gama * torch.log(P))).sum()