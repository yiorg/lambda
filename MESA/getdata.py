import pandas as pd
import os
import scipy.stats.mstats as mstats
import numpy as np
import matplotlib.pyplot as plt
# NOTE: Given MESA data, run this from the MESA directory


def mode(x):
    """
    Comutes a single value for the mode of an array
    :param x:
    :return: returns the mode of an array
    """
    return mstats.mode(x, axis=None)[0]


def main():
    """
    Syncs up the MESA PSG and Actigraphy data based on the data in the overlap file, saves the results to csv

    """
    # define paths to file directories
    psg_fp = os.path.abspath('mesa-commercial-use/polysomnography/annotations-rpoints')
    act_fp = os.path.abspath('mesa-commercial-use/actigraphy')
    over_fp = os.path.abspath('mesa-commercial-use/overlap/mesa-actigraphy-psg-overlap.csv')

    # get files
    psg_files = os.listdir(psg_fp)
    act_files = os.listdir(act_fp)

    # get shared subjects
    over_subs = pd.read_csv(over_fp, usecols=['mesaid'])['mesaid'].values
    psg_subs = sorted([f[11:15] for f in psg_files if '.csv' in f])
    act_subs = sorted([f[11:15] for f in act_files if '.csv' in f])
    subs = sorted([s for s in act_subs if s in psg_subs and int(s) in over_subs])

    error = []
    for sub in subs:
        print "Processing Subject: {}".format(sub)
        # Actigraphy: get interesting columns
        columns = ['mesaid', 'line', 'linetime', 'activity', 'wake']
        act = pd.read_csv(act_fp + "/mesa-sleep-{}.csv".format(sub), usecols=columns)

        # PSG: get interesting columns,
        columns = ['epoch', 'seconds', 'stage']
        psg = pd.read_csv(psg_fp + "/mesa-sleep-{}-rpoint.csv".format(sub), usecols=columns)

        # PSG: collapse data into 30 second epochs, sleep stage per epoch determined by mode of sleep stages during that epoch
        psg = psg.groupby('epoch')['stage'].apply(lambda x: mode(x)[0])

        # PSG: Convert sleep staging data into binary sleep/wake data with sleep coded to 0, wake to 1
        psg = pd.DataFrame(psg)
        psg.columns = ['psg wake']
        psg = psg.reindex(range(1, psg.index[-1]), method="ffill")  # some epochs are missing, ffill them
        psg[psg['psg wake'] > 0] = 'sleep'
        psg[psg['psg wake'] == 0] = 1
        psg[psg['psg wake'] == 'sleep'] = 0

        # Overlap of actigraphy and psg
        overlap = pd.read_csv(over_fp)

        # Get the overlapping data, slice, concatenate, and save it to file
        start = overlap[overlap['mesaid'] == int(sub)]['line'].values[0]-1  # get the starting point of the overlap
        end = start + len(psg)  # end is just the starting point plus however long the psg lasts
        psg.index += start - 1  # line up the indices
        act = act.iloc[start:end]  # slice the actigraphy data to psg data length
        out = pd.concat([act, psg['psg wake']], axis=1)  # concatenate the sleep/wake data from psg to the actigraphy DF
        out.set_index('linetime', inplace=True)  # set the time as the index to make future processing simple
        out = out[['mesaid', 'activity', 'wake', 'psg wake']]  # save the minimum number of columns for efficiency
        out.replace('', 0, inplace=True)
        out.replace(np.nan, 0, inplace=True)

        # # check data for alignment
        # fig, axes = plt.subplots(nrows=3, ncols=1)
        # out['activity'].plot(ax=axes[0])
        # out['wake'].plot.area(ax=axes[1])
        # out['psg wake'].plot.area(ax=axes[2])
        # plt.show()

        # get the performance of the actigraph predictions from the MESA data
        error.append(np.sum(np.abs(out['psg wake']-out['wake']))/len(out['psg wake']))
        if not os.path.exists("mesa-commercial-use/synced/"):  # make sure there's a directory to save synced data to
            os.makedirs('mesa-commercial-use/synced/')
        out.to_csv('mesa-commercial-use/synced/mesa-sleep-synced-{}.csv'.format(sub))  # save to file
    error = np.mean(np.array(error))
    accuracy = 1.0 - error
    print accuracy


if __name__ == '__main__':
    main()
