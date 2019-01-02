import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from mesa_utils import Generator
from keras.models import Sequential, load_model
from keras import layers
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import LSTM


data_dir = 'mesa-commercial-use/synced/'
file_list = sorted(filter(lambda x: '.csv' in x, os.listdir(data_dir)))
look_back = 20
step = 1
delay = 0
batch_size = 128
input_cols = 1
num_train_subs = 50
num_test_subs = 10
epochs = 10
nodes = 32
steps_per_epoch = 100
enable_plots = False
optimizer = 'adam'
loss = 'mean_squared_error'
metrics = ['accuracy']


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
        max_val_ind = len(gen.data)
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
print "Average accuracy: {}".format(acc)
print "Average loss: {}".format(loss)
curr_time = str(datetime.datetime.now())[:-7]
model.save("model {}.h5".format(curr_time))
