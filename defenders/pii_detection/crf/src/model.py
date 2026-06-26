from defenders.pii_detection.crf.src.train import train
from defenders.pii_detection.crf.src.virterbi import decode_using_viterbi 
import json

class LinearCRF:
    def __init__(self, feature_manager, labels, lr=0.05, epochs=100, l2=0.0001):
        self.feature_manager = feature_manager
        self.labels = labels
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.trained_weights = None

    def get_weights(self):
        return self.trained_weights

    def save_model(self, file_path):
        model_data = {
            "weights": self.trained_weights,
            "labels": self.labels
        }
        with open(file_path, 'w') as f:
            json.dump(model_data, f)

    def load_model(self, file_path):
        with open(file_path, 'r') as f:
            model_data = json.load(f)
            self.trained_weights = model_data["weights"]
            self.labels = model_data["labels"]

    def fit(self, train_dataset,validation_data=None):
        self.trained_weights=train(train_dataset,self.feature_manager,self.labels,
                                  lr=self.lr,epochs=self.epochs,l2=self.l2,validation_data=validation_data)
        
    def predict(self, X):
        prediction, _ = decode_using_viterbi(X,self.feature_manager,self.trained_weights,self.labels)
        return prediction
    
    def predict_multiple_sequences(self, sequences):
        predictions = []
        for seq in sequences:
            pred = self.predict(seq)
            predictions.append(pred)
        return predictions
    