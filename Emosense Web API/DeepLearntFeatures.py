import torch
from fairseq.models.wav2vec import Wav2VecModel
import librosa
import pickle

cp = torch.load('wav2vec_large.pt',map_location=torch.device('cpu'))
model = Wav2VecModel.build_model(cp['args'], task=None)
model.load_state_dict(cp['model'])
model.eval()

def featureExtraction(file):
  y, sr = librosa.load(file,sr=16000)  # y -> (t)
  b = torch.from_numpy(y).unsqueeze(0)    # b -> (1, t)
  z = model.feature_extractor(b)          # z -> (1, 512, t)
  z = model.feature_aggregator(z)         # z -> (1, 512, t)

  return z

def featureMean(file):
  z = featureExtraction(file)
  z = torch.mean(z,2).squeeze(0)          # z -> (1, 512) -> (512)
  return z.tolist()

def feature20BinMeans(file):
  z = featureExtraction(file) 
  #z = torch.mean(z,2).squeeze(0)          # z -> (1, 512) -> (512)
  z = z.squeeze(0)

  start = 0
  end = 0
  a = []


  hop = int(z.size()[1]/20)

  if hop==0:
      print(j)
      return 0
  for k in range(20):
      end = end + hop
      d = z[:,start:end]
      d = torch.mean(d,dim=1)
      a.append(d.tolist())
      start = end
  return a

