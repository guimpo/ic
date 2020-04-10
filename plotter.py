import matplotlib.pyplot as plt
import datetime
import numpy as np

def my_plot(i, eqm_hist):
    now = datetime.datetime.now()
    date = datetime.datetime.now()
    plt.autoscale(tight=True)
    plt.ticklabel_format(useOffset=False)
    eqm_hist = np.array(eqm_hist)
    plt.plot(eqm_hist)
    plt.ylabel('EQM')

    plt.xlabel('Épocas')
    plt.savefig('data/exp-{}-{}-{}-{}-{}-{}-{}-eqm-plot.png'.format(i +
                                                                    1, now.year, now.month, now.day, now.hour, now.minute, now.second))
    plt.clf()
