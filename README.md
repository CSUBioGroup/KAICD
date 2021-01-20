# KAICD
A knowledge attention-based deep learning framework for automatic ICD coding. 
# Usage
## 1. How to preprocess the raw data
Firstly, you need to import the package. 
```python
from utils import *
```
Then you can instance the MIMIC3 object and do preprocessing.
```python
mimic = MIMIC(ADMISSIONSPATH="ADMISSIONS.csv", NOTEEVENTSPATH="NOTEEVENTS.csv", 
              DIAGNOSESICDPATH="DIAGNOSES_ICD.csv", DICDDIAGNOSESPATH="D_ICD_DIAGNOSES.csv")
mimic.get_basic_data(outPath='mimic3/data.csv', noteOutPath='mimic3/note_corpus.txt', titleOutPath='mimic3/title_corpus.txt')
```
> **ADMISSIONSPATH** is the path of "ADMISSIONS.csv" in MIMIC III.  
> **NOTEEVENTSPATH** is the path of "NOTEEVENTS.csv" in MIMIC III.  
> **DIAGNOSESICDPATH** is the path of "DIAGNOSES_ICD.csv" in MIMIC III.  
> **DICDDIAGNOSESPATH** is the path of "D_ICD_DIAGNOSES.csv" in MIMIC III.  
> **outPath** is the output path of preprocessed data.  
> **noteOutPath** is the output path of clinck notes corpus which will be used to train note embedding.  
> **titleOutPath** is the output path of ICD titles corpus which will be used to train title embedding.  

After doing this, you can get 3 new files: ***data.csv***, ***note_corpus.txt***, ***title_corpus.txt***. 

## 2. How to train the model
Firstly, you need to import the package.
```python
from utils import *
from model import *
```
Then you need to instance the data utils class and get the pretrained word embedding. 
```python
dataClass = DataClass(dataPath='mimic3/data.csv', dicddiagnosesPath='D_ICD_DIAGNOSES.csv', noteMaxLen=768)
dataClass.vectorize(noteFeaSize=320, titleFeaSize=192, iters=10, 
                    noteCorpusPath=‘mimic3/note_corpus.txt’, loadCache=True)
```
> **dataPath** is the path of "data.csv".  
> **dicddiagnosesPath** is the path of "D_ICD_DIAGNOSES.csv".  
> **noteMaxLen** is the maximum number of keeped words in clinic notes. The words with low TF-IDF will be removed.  
> **noteFeaSize** is the embedding size of clinic notes.  
> **titleFeaSize** is the embedding size of ICD titles.  
> **iters** is the number of iterations in pretraining word embedding.  

Finally, you can instance the model class and train the model. 
```python
model = KAICD(dataClass.classNum, dataClass.vector['noteEmbedding'], dataClass.vector['titleEmbedding'], dataClass.tokenizedTitle, dataClass.titleLen,
              noteFeaSize=320, titleFeaSize=192, hiddenSize=128, filterSize=448,
              embDropout=0.0, fcDropout=0.5)
model.train(dataClass, trainSize=128, batchSize=256, epoch=1000,
            threshold=0.2, earlyStop=30,
            savePath='model/KAICD', 
            metrics="MiF", report=["MiF", "MiAUC"])
```
> **noteFeaSize** is the embedding size of clinic notes.   
> **titleFeaSize** is the embedding size of ICD titles.  
> **hiddenSize** is the hidden size of the BiRNN.  
> **filterSize** is the filter size of the CNN.  
> **embDropout**  is the dropout rate after embedding layer.  
> **fcDropout** is the dropout rate before output layer.  
> **threshold** is the threshold in multi label prediction.  
> **report** is a list of reported indicators after each epoch.  

## 3. How to do prediction
```python
model.load(path="xxx.pkl", map_location="cpu", dataClass=dataClass)
model.to_eval_mode()
Y_prob_pre,Y = model.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(batchSize=64, type='valid', device=model.device))
```
**path** is the save path of your trained model. 
