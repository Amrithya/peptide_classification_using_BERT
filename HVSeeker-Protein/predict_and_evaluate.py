from protein_bert.proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from protein_bert.proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs, GlobalAttention
from protein_bert.proteinbert.finetuning import filter_dataset_by_len, split_dataset_by_len, encode_dataset, get_evaluation_results
from protein_bert.proteinbert import load_pretrained_model, InputEncoder
from protein_bert.proteinbert import conv_and_global_attention_model 
import numpy as np
from protein_bert.proteinbert.model_generation import load_pretrained_model_from_dump
import random
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve
import os
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import pickle
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

def predict_and_evaluate(test_file, modelpath, output_dir):
    # Load the pretrained model and input encoder
    pretrained_model_generator, input_encoder = load_pretrained_model()
    
    # Define output specifications
    OUTPUT_TYPE = OutputType(False, 'binary')
    UNIQUE_LABELS = [0, 1]
    OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)
    
    # Load the test dataset
    test_set = pd.read_csv(test_file).dropna().drop_duplicates()
    test_set.columns = ["label", "seq"]
    
    # Load the model weights
    with open(modelpath, 'rb') as pickle_file:
        modelinfos = pickle.load(pickle_file)
    
    # Create the model
    model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs, dropout_rate=0.6930443135406078)
    model = model_generator.create_model(seq_len=573)
    model.set_weights(modelinfos)
    model_generator.update_state(model)
    model_generator.optimizer_weights = None
    
    # Encode the test dataset
    X_val, y_val, _ = encode_dataset(test_set['seq'], test_set['label'], input_encoder, OUTPUT_SPEC, seq_len=573, needs_filtering=False)
    
    # Make predictions
    y_pred = model.predict(X_val, batch_size=32)
    y_pred = y_pred.flatten()
    y_true = y_val.flatten()
    
    # Calculate evaluation metrics
    f1 = f1_score(y_true, [1 if a > 0.5 else 0 for a in y_pred])
    recall = recall_score(y_true, [1 if a > 0.5 else 0 for a in y_pred])
    prec = precision_score(y_true, [1 if a > 0.5 else 0 for a in y_pred])
    acc = accuracy_score(y_true, [1 if a > 0.5 else 0 for a in y_pred])
    
    # Print metrics
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")
    print(f"Precision: {prec}")
    print(f"Accuracy: {acc}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.show()
    
    # Save predictions and true labels to a CSV file
    dict_ = {"sequence": test_set['seq'], "y_true": y_true, "y_pred": y_pred}
    df = pd.DataFrame(dict_)
    df.to_csv(f"{output_dir}/predictions.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict and evaluate using a pre-trained model.')
    parser.add_argument('-t', '--test_file', required=True, help='Path to the test file.')
    parser.add_argument('-m', '--modelpath', required=True, help='Path to the model file.')
    parser.add_argument('-o', '--output_path', required=True, help='Directory to save output files.')
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    
    # Run prediction and evaluation
    predict_and_evaluate(args.test_file, args.modelpath, args.output_path)