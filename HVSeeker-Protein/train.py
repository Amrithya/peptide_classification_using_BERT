import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score, recall_score
from tensorflow import keras
from sklearn.model_selection import train_test_split
from protein_bert.proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from protein_bert.proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs


def train(arguments, train_file, test_file, savefile):
    max_epochs_per_stage, dropout_rate, patience1, patience2, factor, min_lr, lr, lr_with_frozen_pretrained_layers = arguments

    OUTPUT_TYPE = OutputType(False, 'binary')
    UNIQUE_LABELS = [0, 1]
    OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)
    
    full_set = pd.read_csv(train_file).dropna().drop_duplicates()
    full_set.columns = ["label", "seq"]
    
    test_set = pd.read_csv(test_file).dropna().drop_duplicates()
    test_set.columns = ["label", "seq"]
    
    train_set, valid_set = train_test_split(full_set, stratify=full_set['label'], test_size=0.1, random_state=0)
    
    pretrained_model_generator, input_encoder = load_pretrained_model()
    model_generator = FinetuningModelGenerator(
        pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs, dropout_rate=dropout_rate
    )
    
    training_callbacks = [
        keras.callbacks.ReduceLROnPlateau(patience=patience1, factor=factor, min_lr=min_lr, verbose=1),
        keras.callbacks.EarlyStopping(patience=patience2, restore_best_weights=True)
    ]
    
    history = finetune(
        model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'], train_set['label'], valid_set['seq'], valid_set['label'],
        seq_len=573, batch_size=32, max_epochs_per_stage=max_epochs_per_stage, lr=lr, begin_with_frozen_pretrained_layers=True,
        lr_with_frozen_pretrained_layers=lr_with_frozen_pretrained_layers, n_final_epochs=1, final_seq_len=1500, final_lr=min_lr,
        callbacks=training_callbacks
    )
    
    if history is None:
        print("Training did not return a history object. Skipping history-related operations.")
        return
    
    results, confusion_matrix = evaluate_by_len(
        model_generator, input_encoder, OUTPUT_SPEC, test_set['seq'], test_set['label'],
        start_seq_len=573, start_batch_size=32
    )
    
    model = model_generator.create_model(seq_len=573)
    with open(savefile + "/model.pkl", 'wb') as f:
        pickle.dump(model.get_weights(), f)
    
    # Save F1-score, accuracy, and sensitivity
    metrics = []
    for epoch, logs in enumerate(history.history['val_loss']):
        preds = model.predict(valid_set['seq'])
        preds_binary = (preds > 0.5).astype(int)
        
        f1 = f1_score(valid_set['label'], preds_binary)
        acc = accuracy_score(valid_set['label'], preds_binary)
        sensitivity = recall_score(valid_set['label'], preds_binary)
        
        metrics.append([epoch+1, f1, acc, sensitivity])
    
    metrics_df = pd.DataFrame(metrics, columns=['Epoch', 'F1-score', 'Accuracy', 'Sensitivity'])
    metrics_df.to_csv(savefile + '/metrics.csv', index=False)
    
    # Plot F1-score vs. Epoch
    plt.figure()
    plt.plot(metrics_df['Epoch'], metrics_df['F1-score'], marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.title('F1-score vs. Epoch')
    plt.savefig(savefile + '/f1_vs_epoch.png')
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(valid_set['label'], preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(savefile + '/roc_curve.png')
    
    return
    

if __name__ == "__main__":
    cmdline_parser = argparse.ArgumentParser('training')
    cmdline_parser.add_argument('-t', '--test_file', default='./test_file.csv', help='name of test file', type=str)
    cmdline_parser.add_argument('-f', '--train_file', default='./train_file.csv', help='name of train file', type=str)
    cmdline_parser.add_argument('-s', '--save', default='./models', help='path to save the models', type=str)
    
    args, unknowns = cmdline_parser.parse_known_args()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.__version__)
    
    random.seed(31)
    np.random.seed(31)
    tf.random.set_seed(31)
    
    train([3, 0.75, 3, 2, 0.397, 3.86e-05, 0.00035, 0.00112], args.train_file, args.test_file, args.save)