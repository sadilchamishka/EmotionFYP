from flask import Flask, request, jsonify
from UtterenceModel import predictUtterence
from DeepLearntFeatures import featureMean,feature20BinMeans
from ConversationModel import predictConversationOffline, predictConversationOnline
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

utterence_folder = './utterences/'

@app.route("/")
def home():
		return "success"

@app.route("/utterence",methods = ['POST'])
def utterenceEmotionPrediction():
		file = request.files['audio']
		utterence_path = utterence_folder+'utt.wav'
		file.save(utterence_path)
		prediction = predictUtterence(utterence_path)
		return jsonify({'prediction': prediction[0]})

@app.route("/conversation/offline",methods = ['POST'])
def conversationEmotionPredictionOffline():
		files = request.files
		data = request.args['speakers']	
		prediction,attention_f,attention_b = predictConversationOffline(files,data)
		
		emotion_predictions = []
		attentions = []
		
		i=0
		for p, q, r in zip(prediction.tolist(), attention_f[0][0], attention_b[0][0]):
			temp = {}
			att = {}

			temp['timestep'] = i
			temp['Happy'] = p[0]
			temp['Sad'] = p[1]
			temp['Neutral'] = p[2]
			temp['Angry'] = p[3]
			temp['Excited'] = p[4]
			temp['Frustrated'] = p[5]

			att['timestep'] = i
			att['Forward'] = q
			att['Backward'] = r

			emotion_predictions.append(temp)
			attentions.append(att)

			i+=1

		#return jsonify({'prediction': prediction.tolist(), 'attentionf':attention_f, 'attentionb':attention_b})
		return jsonify({'prediction':emotion_predictions, 'attention':attentions}) 

@app.route("/conversation/online",methods = ['POST'])
def conversationEmotionPredictionOnline():
		files = request.files
		data = request.args['speakers']	
		prediction = predictConversationOnline(files,data)
		print(prediction)
		return "success"

if __name__ == "__main__":
		app.run(host='0.0.0.0')
