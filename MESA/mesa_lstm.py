import os
import json
import numpy as np
import datetime
from mesa_utils import Generator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import random
random.seed(1)


def run(data_dir, file_list, params, save=False):
    # params
    look_back = params['look back']     # how many steps to look back from current point
    step = params['step']       # number of steps between samples
    delay = params['delay']     # delay between current time and the time of the predicted label
    batch_size = params['batch size']       # number of samples in a batch
    input_cols = params['input columns']        # number of input columns
    num_train_subs = params['number of training subjects']      # number of subjects to train on
    num_test_subs = params['number of testing subjects']        # number of subjects to test on
    epochs = params['epochs']       # number of epochs
    nodes = params['nodes']     # number of nodes in the network
    steps_per_epoch = params['steps per epoch']     # number of steps per epoch
    optimizer = params['optimizer']     # kind of optimizer
    loss = params['loss']       # loss function
    metrics = params['metrics']     # metrics to be recorded while training/testing

    # MODEL DEFINITION
    model = Sequential()
    model.add(LSTM(nodes, input_shape=(look_back // step, input_cols)))
    model.add(Dense(1))
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    # RUN
    metrics_list = []
    for i, f in enumerate(file_list):

        # RUN TRAINING SUBJECTS
        if i <= num_train_subs:
            f_name = os.path.join(data_dir, f)
            gen = Generator(f_name, source='MESA')
            train_gen = gen.flow(look_back=look_back,
                                 delay=delay,
                                 min_index=0,
                                 max_index=None,
                                 step=step,
                                 batch_size=batch_size)

            model.fit_generator(train_gen,
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs,
                                verbose=0)

        # RUN TESTING SUBJECTS
        elif i <= num_train_subs + num_test_subs:
            f_name = os.path.join(data_dir, f)
            gen = Generator(f_name, source='MESA')
            test_gen = gen.flow(look_back=look_back,
                                delay=delay,
                                min_index=0,
                                max_index=None,
                                step=step,
                                batch_size=batch_size)

            metrics_list.append(model.evaluate_generator(test_gen, len(gen.data)-1-look_back, verbose=False))

        # FINISH
        else:
            break

    # GET METRICS
    metrics_list = np.array(metrics_list)
    average_metrics = metrics_list.mean(axis=0)
    acc = average_metrics[1]
    loss = average_metrics[0]
    if save:
        curr_time = str(datetime.datetime.now())[:-7]
        # save configuration
        os.mkdir(curr_time)
        model.save("{}/model.h5".format(curr_time))
        with open('{}/params.txt'.format(curr_time), 'w+') as f:
            f.write(json.dumps(params))
        with open('{}/results.txt'.format(curr_time), 'w+') as f:
            f.write("Average accuracy: {}\n".format(acc))
            f.write("Average loss: {}\n".format(loss))
    return acc, loss


if __name__ == '__main__':

    # SETUP
    data = 'mesa-commercial-use/synced/'
    files = sorted(filter(lambda x: '.csv' in x, os.listdir(data)))
    random.shuffle(files)
    parameters = {'look back': 20,
                  'step': 1,
                  'delay': -10,
                  'batch size': 128,
                  'input columns': 1,
                  'number of training subjects': 130,
                  'number of testing subjects': 30,
                  'epochs': 20,
                  'nodes': 10,
                  'steps per epoch': 100,
                  'optimizer': 'adam',
                  'loss': 'mse',
                  'metrics': ['accuracy']}

    res = []
    for lb in range(10, 26, 2):
        for nd in range(8, 100, 8):
            parameters['nodes'] = nd
            parameters['look back'] = lb
            parameters['delay'] = -1 * lb/2
            ac, los = run(data, files, parameters)
            res.append([ac, los, ('look back', lb), ('delay', lb/2), ('nodes', nd)])
    # save results
    curr_time = str(datetime.datetime.now())[:-7]
    os.mkdir(curr_time)
    with open('{}/results.txt'.format(curr_time), 'w+') as f:
        for line in res:
            f.write(str(line)[1:-1] + "\n")
