from __future__ import division
import numpy as np
import pandas
import matplotlib.pyplot as plt

timing_file = pandas.read_csv('/data/tethys/Documents/Projects/NMoon_TTVs/KOI_transit_timing_uncertainties.csv')

periods = np.array(timing_file['koi_period'])
timings = np.array(timing_file['koi_time0bk'])
timing_err_up = np.array(timing_file['koi_time0bk_err1'])
timing_err_down = np.array(timing_file['koi_time0bk_err2'])
timing_err_average = (np.abs(timing_err_up) + np.abs(timing_err_down)) / 2

fractional_timing_err = timing_err_average / timings 

#### plot this is a function of orbital period -- presumably, shorter period correlates with lower fractional duration error

plt.scatter(periods, timing_err_average, s=30, alpha=0.5, color='DodgerBlue', edgecolor='k')
plt.xlabel('Period [days]')
plt.ylabel('Timing Error')
plt.show()