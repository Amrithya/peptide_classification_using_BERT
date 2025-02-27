import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.__version__)
import pickle
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from protein_bert.proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from protein_bert.proteinbert import load_pretrained_model, InputEncoder
from protein_bert.proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
import numpy as np
import random
import argparse



def train(arguments, train_file, test_file, savefile):

        max_epochs_per_stage = arguments[0] 
        dropout_rate = arguments[1] 
        patience1 = arguments[2] 
        patience2 = arguments[3] 
        factor = arguments[4] 
        min_lr = arguments[5] 
        lr = arguments[6] 
        lr_with_frozen_pretrained_layers = arguments[7]
        file_ = train_file

        OUTPUT_TYPE = OutputType(False, 'binary')
        UNIQUE_LABELS = [0, 1]
        OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)
        
        full_set = pd.read_csv(file_).dropna().drop_duplicates()
        full_set.columns = ["label","seq"]

        test_set = pd.read_csv(test_file).dropna().drop_duplicates()
        test_set.columns = ["label","seq"]
        
        train_set, valid_set = train_test_split(full_set, stratify = full_set['label'], test_size = 0.1, random_state = 0)

        pretrained_model_generator, input_encoder = load_pretrained_model()
        model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function = \
        get_model_with_hidden_layers_as_outputs, dropout_rate = dropout_rate)

        training_callbacks = [
        keras.callbacks.ReduceLROnPlateau(patience = patience1, factor = factor, min_lr = min_lr, verbose = 1),
        keras.callbacks.EarlyStopping(patience = patience2, restore_best_weights = True),
        ]

        finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'], train_set['label'], valid_set['seq'], valid_set['label'], \
        seq_len = 573, batch_size = 32, max_epochs_per_stage = max_epochs_per_stage, lr = lr, begin_with_frozen_pretrained_layers = True, \
        lr_with_frozen_pretrained_layers = lr_with_frozen_pretrained_layers, n_final_epochs = 1, final_seq_len = 1500, final_lr = min_lr, callbacks = training_callbacks)


        results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, test_set['seq'], test_set['label'], \
        start_seq_len = 573, start_batch_size = 32)


        model = model_generator.create_model(seq_len=573)



        with open(savefile + "/model.pkl", 'wb') as f:
            pickle.dump((model.get_weights()), f)
  
        return
        
        
if __name__ == "__main__":

    cmdline_parser = argparse.ArgumentParser('training')



    cmdline_parser.add_argument('-t', '--test_file',
                                default='./test_file.csv',
                                help='name of test file',
                                type=str)
    cmdline_parser.add_argument('-f', '--train_file',   
                                default='./train_file.csv',
                                help='name of train file',
                                type=str)                            
    cmdline_parser.add_argument('-s', '--save',
                                default='./models',
                                help='path to save the models',
                                type=str)

    args, unknowns = cmdline_parser.parse_known_args()     

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print(tf.__version__)
    random.seed(31)
    #keras.utils.set_random_seed(31)
    np.random.seed(31)
    tf.random.set_seed(31)

    train([5, 0.6930443135406078, 3, 2, 0.39657786522163363, 3.8602329788847605e-05, 0.00035016815751852485, 0.0011187914605146963], args.train_file, args.test_file, args.save)



