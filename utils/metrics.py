from sklearn.metrics import f1_score, accuracy_score

def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    # macro-F1 is critical for HAR datasets with class imbalance
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return acc, macro_f1