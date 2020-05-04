import torch,time,os,pickle
from torch import nn
from torch.nn import functional as F
import numpy as np
from collections import *
from nnLayer import *
from metrics import *

class BaseModel:
    def __init__(self):
        pass
    def calculate_y_logit(self):
        pass
    def train(self, dataClass, trainSize, batchSize, epoch, 
              lr=0.001, stopRounds=10, threshold=0.2, earlyStop=10, 
              savePath='model/KAICD', saveRounds=1, isHigherBetter=True, metrics="MiF", report=["ACC", "MiF"]):
        assert batchSize%trainSize==0
        metrictor = Metrictor(dataClass.classNum)
        self.stepCounter = 0
        self.stepUpdate = batchSize//trainSize
        optimizer = torch.optim.Adam(self.moduleList.parameters(), lr=lr)
        trainStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='train', device=self.device)
        itersPerEpoch = (dataClass.trainSampleNum+trainSize-1)//trainSize
        mtc,bestMtc,stopSteps = 0.0,0.0,0
        if dataClass.validSampleNum>0: validStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='valid', device=self.device)
        st = time.time()
        for e in range(epoch):
            for i in range(itersPerEpoch):
                self.to_train_mode()
                X, Y = next(trainStream)
                loss = self._train_step(X, Y, optimizer)
                if stopRounds>0 and (e*itersPerEpoch+i+1)%stopRounds==0:
                    self.to_eval_mode()
                    print("After iters %d: [train] loss= %.3f;"%(e*itersPerEpoch+i+1,loss), end='')
                    if dataClass.validSampleNum>0:
                        X, Y = next(validStream)
                        loss = self.calculate_loss(X,Y)
                        print(' [valid] loss= %.3f;'%loss, end='')
                    restNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*trainSize
                    speed = (e*itersPerEpoch+i+1)*trainSize/(time.time()-st)
                    print(" speed: %.3lf items/s; remaining time: %.3lfs;"%(speed, restNum/speed))
            if dataClass.validSampleNum>0 and (e+1)%saveRounds==0:
                self.to_eval_mode()
                print('========== Epoch:%5d =========='%(e+1))
                print('[Total Train]', end='')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
                metrictor.set_data(Y_pre, Y, threshold)
                metrictor(report)
                print('[Total Valid]', end='')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
                metrictor.set_data(Y_pre, Y, threshold)
                res = metrictor(report)
                mtc = res[metrics]
                print('=================================')
                if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                    print('Bingo!!! Get a better Model with val %s: %.3f!!!'%(metrics,mtc))
                    bestMtc = mtc
                    self.save("%s.pkl"%savePath, e+1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps>=earlyStop:
                        print('The val %s has not improved for more than %d steps in epoch %d, stop training.'%(metrics,earlyStop,e+1))
                        break
        self.load("%s.pkl"%savePath, dataClass=dataClass)
        os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, ("%.3lf"%bestMtc)[2:]))
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
        metrictor.set_data(Y_pre, Y, threshold)
        print('[Total Train]',end='')
        metrictor(report)
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
        metrictor.set_data(Y_pre, Y, threshold)
        print('[Total Valid]',end='')
        res = metrictor(report)
        #metrictor.each_class_indictor_show(dataClass.id2lab)
        print('================================')
        return res
    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            stateDict['nword2id'],stateDict['tword2id'] = dataClass.nword2id,dataClass.tword2id
            stateDict['id2nword'],stateDict['id2tword'] = dataClass.id2nword,dataClass.id2tword
            stateDict['icd2id'],stateDict['id2icd'] = dataClass.id2icd,dataClass.icd2id
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            dataClass.trainIdList = parameters['trainIdList']
            dataClass.validIdList = parameters['validIdList']
            dataClass.testIdList = parameters['testIdList']

            dataClass.nword2id,dataClass.tword2id = parameters['nword2id'],parameters['tword2id']
            dataClass.id2nword,dataClass.id2tword = parameters['id2nword'],parameters['id2tword']
            dataClass.id2icd,dataClass.icd2id = parameters['icd2id'],parameters['id2icd']     
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.sigmoid(Y_pre)
    def calculate_y(self, X, threshold=0.2):
        Y_pre = self.calculate_y_prob(X)
        isONE = Y_pre>threshold
        Y_pre[isONE],Y_pre[~isONE] = 1,0
        return Y_pre
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        return self.crition(input=Y_logit, target=Y)
    def calculate_indicator_by_iterator(self, dataStream, classNum, report, threshold):
        metrictor = Metrictor(classNum)
        Y_prob_pre,Y = self.calculate_y_prob_by_iterator(dataStream)
        Metrictor.set_data(Y_prob_pre, Y, threshold)
        return metrictor(report)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr
    def calculate_y_by_iterator(self, dataStream, threshold=0.2):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        isONE = Y_preArr>threshold
        Y_preArr[isONE],Y_preArr[~isONE] = 1,0
        return Y_preArr, YArr
    def to_train_mode(self):
        for module in self.moduleList:
            module.train()
    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()
    def _train_step(self, X, Y, optimizer):
        self.stepCounter += 1
        if self.stepCounter<self.stepUpdate:
            p = False
        else:
            self.stepCounter = 0
            p = True
        loss = self.calculate_loss(X, Y)/self.stepUpdate
        loss.backward()
        if p:
            optimizer.step()
            optimizer.zero_grad()
        return loss*self.stepUpdate

class KAICD(BaseModel):
    def __init__(self, classNum, noteEmbedding, titleEmbedding, tokenizedTitle, titleLen,
                 noteFeaSize=256, titleFeaSize=256, hiddenSize=64, filterSize=64, contextSizeList=[1,3,5], hiddenSizeList=[],
                 embDropout=0.2, fcDropout=0.5, isNoteEmbFreezon=False, isTitleEmbFreezon=True, device=torch.device("cuda:0")):
        self.titles,self.titleLens = torch.tensor(tokenizedTitle,dtype=torch.long,device=device),torch.tensor(titleLen,dtype=torch.long,device=device)
        self.noteEmbedding = TextEmbedding(noteEmbedding, freeze=isNoteEmbFreezon, dropout=embDropout, name='noteEmbedding').to(device)
        self.titleEmbedding = TextEmbedding(titleEmbedding, freeze=isTitleEmbFreezon, dropout=embDropout, name='titleEmbedding').to(device)
        self.textCNN = TextCNN(noteFeaSize, filterSize, contextSizeList).to(device)
        self.textBiRNN = TextGRU(titleFeaSize, hiddenSize, bidirectional=True).to(device)
        self.simpleAttn = SimpleAttention2(hiddenSize*2, hiddenSize//2).to(device)
        self.knowledgeAttn = KnowledgeAttention(len(contextSizeList)*filterSize, hiddenSize*2).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterSize+hiddenSize*2, classNum, hiddenSizeList, dropout=fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.noteEmbedding,self.textCNN,self.titleEmbedding,self.textBiRNN,self.simpleAttn,self.knowledgeAttn,self.fcLinear])
        self.crition = nn.MultiLabelSoftMarginLoss()
        self.device = device
    def calculate_y_logit(self, input):
        x = input['noteArr']
        # x: batchSize × seqLen; inputLabel: labelNum × seqLen × feaSize
        noteConved = self.textCNN(self.noteEmbedding(x)) # => batchSize × scaleNum*filterSize
        labelEncoded = self.textBiRNN(self.titleEmbedding(self.titles), self.titleLens) # => labelNum × seqLen × hiddenSize*2
        labelEncoded = self.simpleAttn(labelEncoded) # => labelNum × hiddenSize*2

        knowledgeAttned = self.knowledgeAttn(noteConved, labelEncoded) # => batchSize × hiddenSize*2
        return self.fcLinear(torch.cat([knowledgeAttned, noteConved], dim=1)) # => batchSize × classNum

class KAICD_onlyCNN(BaseModel):
    def __init__(self, classNum, noteEmbedding, titleEmbedding, tokenizedTitle, titleLen,
                 noteFeaSize=256, titleFeaSize=256, hiddenSize=64, filterSize=64, contextSizeList=[1,3,5], hiddenSizeList=[],
                 embDropout=0.2, fcDropout=0.5, isNoteEmbFreezon=False, isTitleEmbFreezon=True, device=torch.device("cuda:0")):
        self.titles,self.titleLens = torch.tensor(tokenizedTitle,dtype=torch.long,device=device),torch.tensor(titleLen,dtype=torch.long,device=device)
        self.noteEmbedding = TextEmbedding(noteEmbedding, freeze=isNoteEmbFreezon, dropout=embDropout, name='noteEmbedding').to(device)
        self.titleEmbedding = TextEmbedding(titleEmbedding, freeze=isTitleEmbFreezon, dropout=embDropout, name='titleEmbedding').to(device)
        self.textCNN = TextCNN(noteFeaSize, filterSize, contextSizeList).to(device)
        #self.textBiRNN = TextGRU(titleFeaSize, hiddenSize, bidirectional=True).to(device)
        #self.simpleAttn = SimpleAttention2(hiddenSize*2, hiddenSize//2).to(device)
        #self.knowledgeAttn = KnowledgeAttention(len(contextSizeList)*filterSize, hiddenSize*2).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterSize, classNum, hiddenSizeList, dropout=fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.noteEmbedding,self.textCNN,self.titleEmbedding,self.fcLinear])
        self.crition = nn.MultiLabelSoftMarginLoss()
        self.device = device
    def calculate_y_logit(self, input):
        x = input['noteArr']
        # x: batchSize × seqLen; inputLabel: labelNum × seqLen × feaSize
        noteConved = self.textCNN(self.noteEmbedding(x)) # => batchSize × scaleNum*filterSize
        #labelEncoded = self.textBiRNN(self.titleEmbedding(self.titles), self.titleLens) # => labelNum × seqLen × hiddenSize*2
        #labelEncoded = self.simpleAttn(labelEncoded) # => labelNum × hiddenSize*2

        #knowledgeAttned = self.knowledgeAttn(noteConved, labelEncoded) # => batchSize × hiddenSize*2
        return self.fcLinear(noteConved) # => batchSize × classNum

class ICDLabeler0(BaseModel):
    def __init__(self, classNum, embedding, 
                 charFeaSize=64, filterSize=96, contextSizeList=[1,3,5], 
                 embDropout=0.2, fcDropout=0.0, device=torch.device('cuda:0')):
        self.embedding = TextEmbedding(embedding, dropout=embDropout).to(device)
        self.textCNN = TextCNN(charFeaSize, filterSize, contextSizeList).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterSize, classNum, [], dropout=fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.embedding, self.textCNN, self.fcLinear])
        self.crition = nn.MultiLabelSoftMarginLoss()
        self.device = device
    def calculate_y_logit(self, input):
        x = input['noteArr']
        # x: batchSize × seqLen; 
        x = self.embedding(x) # => batchSize × seqLen × feaSize; 
        x = self.textCNN(x) # => batchSize × scaleNum*filterSize
        return self.fcLinear(x) # => batchSize × classNum
        
class ICDLabeler1(BaseModel):
    def __init__(self, classNum, embedding, L, 
                 charFeaSize=64, seqLen=32, contextSizeList=[1,3,5], hiddenSizeList=[], 
                 embDropout=0.2, fcDropout=0.0, device=torch.device('cuda:0')):
        self.embedding = TextEmbedding(embedding, dropout=embDropout).to(device)
        self.textCNN = TextCNN(charFeaSize, classNum, contextSizeList, reduction='None').to(device)
        self.gcn = GCN(classNum, L, seqLen, seqLen//8, hiddenSizeList, embDropout).to(device)
        self.fcLinear = MLP(seqLen//8, 1, [], dropout=fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.embedding, self.textCNN, self.gcn, self.fcLinear])
        self.crition = nn.MultiLabelSoftMarginLoss()
        self.device = device
    def calculate_y_logit(self, input):
        x = input['noteArr']
        # x: batchSize × seqLen; 
        x = self.embedding(x) # => batchSize × seqLen × feaSize; 
        x = self.textCNN(x) # => scaleNum * (batchSize × seqLen × classNum)
        x1 = (sum(x)/len(x)).transpose(1,2) # => batchSize × classNum × seqLen

        x2 = self.gcn(x1)[-1] # => batchSize × classNum × seqLen//8
        return self.fcLinear(x2).squeeze(2) #  batchSize × classNum

class ICDLabeler2(BaseModel):
    def __init__(self, classNum, embedding, L, 
                 charFeaSize=64, outSize=39, contextSizeList=[1,3,5], hiddenSizeList=[9, 27], 
                 embDropout=0.2, fcDropout=0.0, device=torch.device('cuda:0')):
        self.embedding = TextEmbedding(embedding, dropout=embDropout).to(device)
        self.textCNN = TextCNN(charFeaSize, classNum, contextSizeList, reduction='GCN').to(device)
        self.gcn = GCN(classNum, L, len(contextSizeList), hiddenSizeList, embDropout).to(device)
        self.fcLinear = MLP(outSize, 1, [], dropout=fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.embedding, self.textCNN, self.gcn, self.fcLinear])
        self.crition = nn.MultiLabelSoftMarginLoss()
        self.device = device
    def calculate_y_logit(self, input):
        x = input['noteArr']
        # x: batchSize × seqLen; 
        x = self.embedding(x) # => batchSize × seqLen × feaSize; 
        x = self.textCNN(x) # => batchSize × classNum × scaleNum
        x = self.gcn(x) # => layerNum * (batchSize × classNum × outSize)
        return self.fcLinear(torch.cat(x, dim=2)).squeeze(2) #  batchSize × classNum

class ICDLabeler3(BaseModel):
    def __init__(self, classNum, embedding, L, 
                 charFeaSize=64, filterSize=96, contextSizeList=[1,3,5], hiddenSizeList=[1,1], 
                 embDropout=0.2, fcDropout=0.0, device=torch.device('cuda:0')):
        self.embedding = TextEmbedding(embedding, dropout=embDropout).to(device)
        self.textCNN = TextCNN(charFeaSize, filterSize, contextSizeList).to(device)
        self.gcn = GCN(classNum, L, 1, hiddenSizeList, embDropout).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterSize, classNum, [], dropout=fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.embedding, self.textCNN, self.gcn, self.fcLinear])
        self.crition = nn.MultiLabelSoftMarginLoss()
        self.device = device
    def calculate_y_logit(self, input):
        x = input['noteArr']
        # x: batchSize × seqLen; 
        x = self.embedding(x) # => batchSize × seqLen × feaSize; 
        x = self.textCNN(x) # => batchSize × scaleNum*filterSize
        x0 = self.fcLinear(x).unsqueeze(dim=2) # => batchSize × classNum × 1
        x1 = self.gcn(x0) # => layerNum * (batchSize × classNum × 1)
        return (x0+sum(x1)).squeeze(dim=2) #  batchSize × classNum


class ICDAttentionalCNN(BaseModel):
    def __init__(self, noteEmbedding, classNum, noteFeaSize=256, labFeaSize=256, contextSizeList=[1,3,5], filterSize=64,
                 embDropout=0.2, fcDropout=0.5, isNoteEmbFreezon=False, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding(noteEmbedding, freeze=isNoteEmbFreezon, dropout=embDropout).to(device)
        self.textICDAttnCNN = TextICDAttentionalCNN(noteFeaSize, contextSizeList, filterSize, classNum).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding,self.textICDAttnCNN])
        self.crition = nn.MultiLabelSoftMarginLoss()
        self.device = device

    def calculate_y_logit(self, input):
        x = input['noteArr']
        # x: batchSize × seqLen;
        return self.textICDAttnCNN(self.textEmbedding(x)) # => batchSize × classNum

class ShiModel(BaseModel):
    def __init__(self, noteEmbedding, titleEmbedding, classNum, tokenizedTitle, titleLen, twdLen, 
                 feaSize=50, hiddenSize=200, 
                 embDropout=0.5, fcDropout=0.2, device=torch.device('cuda:0')):
        self.title = torch.tensor(tokenizedTitle, dtype=torch.long, device=device) # classNum × seqLen × wdLen
        self.titleLen = torch.tensor(titleLen, dtype=torch.long, device=device)
        self.twdLen = torch.tensor(twdLen, dtype=torch.long, device=device)
        self.noteEmbedding = TextEmbedding(noteEmbedding, freeze=False, dropout=0.0, name='noteEmbedding').to(device)
        self.titleEmbedding = TextEmbedding(titleEmbedding, freeze=False, dropout=0.0, name='titleEmbedding').to(device)
        self.noteCharLSTM = TextLSTM(feaSize, hiddenSize, False, name='noteCharLSTM').to(device)
        self.noteWordLSTM = TextLSTM(hiddenSize, hiddenSize, False, name='noteWordLSTM').to(device)
        self.titleCharLSTM = TextLSTM(feaSize, hiddenSize, False, name='titleCharLSTM').to(device)
        self.titleWordLSTM = TextLSTM(hiddenSize, hiddenSize, False, name='titleWordLSTM').to(device)
        self.luongAttn = TextAttention('dot').to(device)
        self.embeddingDropout = nn.Dropout(p=embDropout).to(device)
        self.fcLinear = MLP(hiddenSize, 1, dropout=fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.noteEmbedding, self.titleEmbedding, 
                                         self.noteCharLSTM, self.noteWordLSTM, self.titleCharLSTM, self.titleWordLSTM, 
                                         self.luongAttn, self.fcLinear])

        self.crition = nn.MultiLabelSoftMarginLoss()
        self.device = device

    def calculate_y_logit(self, input):
        # Only support batchSize 
        # x: descNum × seqLen × wdLen; xLen: descNum; nwdLen: descNum × seqLen
        x,xLen,nwdLen = input['noteArr'],input['noteLen'],input['wordLen']
        noteVec = self.embeddingDropout(self.encode(self.noteEmbedding, x, xLen, nwdLen, self.noteCharLSTM, self.noteWordLSTM)) # => descNum × hiddenSize
        titleVec = self.embeddingDropout(self.encode(self.titleEmbedding, self.title, self.titleLen, self.twdLen, self.titleCharLSTM, self.titleWordLSTM)) # => classNum × hiddenSize
        attn = self.luongAttn(noteVec, titleVec) # => classNum × hiddenSize
        return self.fcLinear(attn).squeeze(1).unsqueeze(0) # => 1 × classNum

    def encode(self, embedding, x, xLen, wordLen, charLSTM, wordLSTM):
        # x: batchSize × seqLen × wdLen; xLen: batchSize; wordLen: batchSize × seqLen
        x = embedding(x) # => batchSize × seqLen × wdLen × feaSize
        xLen,indices = torch.sort(xLen, descending=True)
        _,desortedIndices = torch.sort(indices)
        packedX = nn.utils.rnn.pack_padded_sequence(x[indices], xLen, batch_first=True) # => batchSize*seqLen × wdLen × feaSize
        packedWordLen = nn.utils.rnn.pack_padded_sequence(wordLen[indices], xLen, batch_first=True) # => batchSize*seqLen
        x = charLSTM(packedX.data, packedWordLen.data)[:,-1,:] # => batchSize*seqLen × hiddenSize
        packedX = nn.utils.rnn.PackedSequence(data=x, 
                                              batch_sizes=packedX.batch_sizes,
                                              sorted_indices=packedX.sorted_indices,
                                              unsorted_indices=packedX.unsorted_indices)
        packedX = wordLSTM(packedX, None) # => batchSize*seqLen × hiddenSize
        x,_ = nn.utils.rnn.pad_packed_sequence(packedX, batch_first=True) # => batchSize × hiddenSize
        x = x[desortedIndices, -1, :]
        return x # batchSize × hiddenSize
