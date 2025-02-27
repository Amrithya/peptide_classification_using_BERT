import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from protein_bert.proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from protein_bert.proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from protein_bert.proteinbert import load_pretrained_model
import numpy as np
import random
from skopt import BayesSearchCV, gp_minimize
import time
from skopt.space import Space, Categorical, Integer, Real, Dimension
import argparse
from functools import partial
import tensorflow as tf

def run_finetuneing(arguments, finetune_file):

        max_epochs_per_stage = arguments[0]
        dropout_rate = arguments[1]
        patience1 = arguments[2]
        patience2 = arguments[3]
        factor = arguments[4]
        min_lr = arguments[5]
        lr = arguments[6]
        lr_with_frozen_pretrained_layers = arguments[7]
        file_ = finetune_file

        OUTPUT_TYPE = OutputType(False, 'binary')
        UNIQUE_LABELS = [0, 1]
        OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)
        result_list = []
        full_set = pd.read_csv(file_).dropna().drop_duplicates()

        for x_range in [1/5,2/5,3/5,4/5,1]:
            
            train_set1 = full_set[int((x_range)*len(full_set)):]
            train_set2 = full_set[:int((x_range-1/5)*len(full_set))]
            
            train_set2 = pd.concat([train_set2, train_set1])

            test_set = full_set[int((x_range-1/5)*len(full_set)):int((x_range)*len(full_set))]
            train_set2.columns = ["label","seq"]
            test_set.columns = ["label","seq"]
            train_set = train_set2
            train_set, valid_set = train_test_split(train_set, stratify = train_set['label'], test_size = 0.1, random_state = 0)
            print(f'{len(train_set)} training set records, {len(valid_set)} validation set records, {len(test_set)} test set records.')

            pretrained_model_generator, input_encoder = load_pretrained_model()


            model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC, pretraining_model_manipulation_function = \
            get_model_with_hidden_layers_as_outputs, dropout_rate = dropout_rate)

            training_callbacks = [
            keras.callbacks.ReduceLROnPlateau(patience = patience1, factor = factor, min_lr = min_lr, verbose = 1),
            keras.callbacks.EarlyStopping(patience = patience2, restore_best_weights = True),
            ]

            finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'], train_set['label'], valid_set['seq'], valid_set['label'], \
            seq_len = 573, batch_size = 32, max_epochs_per_stage = max_epochs_per_stage, lr = lr, begin_with_frozen_pretrained_layers = True, \
            lr_with_frozen_pretrained_layers = lr_with_frozen_pretrained_layers, n_final_epochs = 1, final_seq_len = 573, final_lr = min_lr, callbacks = training_callbacks)

            results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, test_set['seq'], test_set['label'], \
            start_seq_len = 573, start_batch_size = 32)
            result_list.append(1-results["AUC"]["All"])
        

        print(result_list)

    
        return np.mean(result_list)



if __name__ == "__main__":
    
    
    cmdline_parser = argparse.ArgumentParser('optimizer')

    cmdline_parser.add_argument('-o', '--output_file',
                                default="./out_result.npy",
                                help='output_file',
                                type=str)
    cmdline_parser.add_argument('-f', '--finetune_file',
                                default="./finetune_file.csv",
                                help='output_file',
                                type=str)

    args, unknowns = cmdline_parser.parse_known_args()

    random.seed(31)
    tf.random.set_seed(31)
    np.random.seed(31)
    objective_finetune = partial(run_finetuneing, finetune_file=args.finetune_file) 
    start_time = time.time()
    space = [Integer(10,50, name='max_epochs_per_stage'),Real(0.1, 0.7, prior = "uniform", name="dropout_rate"),Integer(1,3, name="partience1"),Integer(1,3, name="partience2"),Real(0.1, 0.4, prior = "uniform", name ="factor"), Real(1e-6,1e-4, prior='log-uniform', name = "min_lr"), Real(1e-5,1e-3, prior='log-uniform', name="lr"),Real(1e-3,1e-2, prior='log-uniform', name = "lr_with_frozen_pretrained_layers")]
    res = gp_minimize(objective_finetune,
                  space,
                  acq_func="EI",      
                  n_calls=25,         
                  n_random_starts=8,
                  #noise=0.1**2,       
                  random_state=1234)
                  
    np.save(args.output_file, res)
    end_time = time.time()
    

    
    

