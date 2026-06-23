from defenders.pii_detection.crf.gradients import compute_gradient
from virterbi import decode_using_viterbi
from collections import defaultdict

def train(dataset, feature_manager,labels,lr,l2,epochs,validation_data=None):
    # set initialy weights to 0
    weights = defaultdict(float)
    for feature_name in feature_manager.names():
        weights[feature_name] = 0.0
    
    val_exist=False if validation_data is not None else False

    for epoch in range(epochs):
        for x_train, y_train in dataset:
            if len(x_train) == 0:
                continue

            gradient = compute_gradient(x_train, y_train, feature_manager, weights, labels, l2)
            
            for feature_name in weights:
                weights[feature_name] += lr * gradient[feature_name]

        # evaluate training accuracy
        correct = 0
        total = 0
        for x_train, y_train in dataset:
            if len(x_train) == 0:
                continue
            predicted_labels, _ = decode_using_viterbi(x_train, feature_manager, weights, labels)
            correct += sum(p == t for p, t in zip(predicted_labels, y_train))
            total += len(y_train)
        accuracy = correct / total
       

        # evaluate validation accuracy
        correct_val = 0
        total_val = 0
        if val_exist:
            for x_val, y_val in validation_data:
                if len(x_val) == 0:
                    continue
                predicted_labels_val, _ = decode_using_viterbi(x_val, feature_manager, weights, labels)
                correct_val += sum(p == t for p, t in zip(predicted_labels_val, y_val))
                total_val += len(y_val)
            val_accuracy = correct_val / total_val
        
        if val_exist:
            print(f"Epoch {epoch+1}/{epochs} ,Training Accuracy:{accuracy} ,Validation Accuracy: {val_accuracy}")
        else:
            print(f"Epoch {epoch+1}/{epochs} ,Training Accuracy:{accuracy}")

    return weights

            