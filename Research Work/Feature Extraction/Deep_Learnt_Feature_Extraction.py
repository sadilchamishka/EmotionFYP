import torch
from fairseq.models.wav2vec import Wav2VecModel
import librosa
import numpy as np
import pickle

cp = torch.load('/content/drive/My Drive/EmotionRNN2/wav2vec_large.pt',map_location=torch.device('cpu'))
model = Wav2VecModel.build_model(cp['args'], task=None)
model.load_state_dict(cp['model'])
model.eval()

data = pickle.load(open('/content/drive/My Drive/Emotion RNN/IEMOCAP_features_raw.pkl','rb'), encoding="latin1")
videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = data

base = '/content/drive/My Drive/FYP/IEMOCAP_full_release/Session'
dataset_for_experiment = {}

for i in videoIDs:
  file = base + i[4] + '/sentences/wav/' + i + '/'
  data = []
  for j in videoIDs[i]:
    y, sr = librosa.load(file+j+'.wav',sr=16000)  # y -> (t)
    b = torch.from_numpy(y).unsqueeze(0)               # b -> (1, t)
    z = model.feature_extractor(b)                     # z -> (1, 512, t)
    z = model.feature_aggregator(z).squeeze(0)         # z -> (1, 512, t) -> (512, t)

    start = 0
    end = 0 
    a = []
    hop = int(z.size()[1]/20)
    if hop==0:
      print(j)
    for k in range(20):
      end = end + hop
      d = z[:,start:end]
      d = torch.mean(d,dim=1)
      a.append(d.tolist())
      start = end
    
    data.append(a)

  dataset_for_experiment[i] = data

with open("/content/drive/My Drive/EmotionRNN2/dataformodel_20_bins.pkl", "wb") as f:
    pickle.dump((dataset_for_experiment,videoLabels,videoSpeakers,trainVid,testVid), f)