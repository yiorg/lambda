import os
import json
import numpy as np
import matplotlib.pyplot as plt
import datetime
from mesa_utils import Generator
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM


data_dir = 'mesa-commercial-use/synced/'
file_list = sorted(filter(lambda x: '.csv' in x, os.listdir(data_dir)))
params = {'look back': 20,
          'step': 1,
          'delay': -10,
          'batch size': 128,
          'input columns': 1,
          'number of training subjects': 1300,
          'number of testing subjects': 300,
          'epochs': 10,
          'nodes': 64,
          'steps per epoch': 100,
          'enable plots': False,
          'optimizer': 'adam',
          'loss': 'mse',
          'metrics': ['accuracy']}

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
enable_plots = params['enable plots']       # plot?
optimizer = params['optimizer']     # kind of optimizer
loss = params['loss']       # loss function
metrics = params['metrics']     # metrics to be recorded while training/testing


print "Defining model..."
model = Sequential()
model.add(LSTM(nodes, input_shape=(look_back // step, input_cols)))
model.add(Dense(1))
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)

metrics_list = []
history = []
for i, f in enumerate(file_list):
    if i < num_train_subs:
        f_name = os.path.join(data_dir, f)
        gen = Generator(f_name, source='MESA')
        max_train_ind = len(gen.data) * 4 // 5
        max_val_ind = len(gen.data) - max((0, delay))
        val_steps = (max_val_ind - max_train_ind + 1 - look_back)
        train_gen = gen.flow(look_back=look_back,
                             delay=delay,
                             min_index=0,
                             max_index=max_train_ind,
                             step=step,
                             batch_size=batch_size)
        val_gen = gen.flow(look_back=look_back,
                           delay=delay,
                           min_index=max_train_ind + 1,
                           max_index=max_val_ind,
                           step=step,
                           batch_size=batch_size)

        print "Running model fit for subject {}...".format(f_name.split('-')[-1][:4])
        history = model.fit_generator(train_gen,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=epochs,
                                      verbose=2,
                                      validation_data=val_gen,
                                      validation_steps=val_steps)
        if enable_plots:
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs = range(1, len(loss) + 1)
            plt.figure()
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()
            plt.show()

    elif i < num_train_subs + num_test_subs:
        f_name = os.path.join(data_dir, f)
        gen = Generator(f_name, source='MESA')
        test_gen = gen.flow(look_back=look_back,
                            delay=delay,
                            min_index=0,
                            max_index=None,
                            step=step,
                            batch_size=batch_size)

        print "Running evaluator for subject {}...".format(f_name.split('-')[-1][:4])
        metrics_list.append(model.evaluate_generator(test_gen, len(gen.data)-1-look_back, verbose=False))
        print "Loss and accuracy: ", metrics_list[-1]
    else:
        break

metrics_list = np.array(metrics_list)
average_metrics = metrics_list.mean(axis=0)
acc = average_metrics[1]
loss = average_metrics[0]
curr_time = str(datetime.datetime.now())[:-7]
print "Average accuracy: {}".format(acc)
print "Average loss: {}".format(loss)

# save configuration
os.mkdir(curr_time)
model.save("{}/model.h5".format(curr_time))
with open('{}/params.txt'.format(curr_time), 'w+') as f:
    f.write(json.dumps(params))
with open('{}/results.txt'.format(curr_time), 'w+') as f:
    f.write("Average accuracy: {}\n".format(acc))
    f.write("Average loss: {}\n".format(loss))
