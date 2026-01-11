import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SentimentAnalysis():
    
    def __init__(self):
        pass

    def load_models(self):
        model_path = os.path.join("artifacts","GRU_Sentiment.keras")
        encoder_path = os.path.join("artifacts","label_encoder.pkl")
        max_len_path = os.path.join("artifacts","RNN_max_len.txt")
        token_path = os.path.join("artifacts","RNN_tokenizer.pkl")

        self.model = load_model(model_path)
        self.tokenize= pickle.load(open(token_path,"rb"))
        self.label_encoder = pickle.load(open(encoder_path,"rb"))
        
        with open(max_len_path, "r") as f:
            self.max_len =int(f.read())
        

    def sentiment_analysis(self, input_text):
        self.load_models()
        seq = self.tokenize.texts_to_sequences([input_text])
        pad_seq = pad_sequences(seq, padding="post", maxlen=self.max_len)

        pred_prob = self.model.predict(pad_seq)
        pred_class =np.argmax(pred_prob,axis=1)[0]
        return self.label_encoder.inverse_transform([pred_class])[0]
    
