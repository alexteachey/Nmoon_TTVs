from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os
import time
import traceback
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit


#### THIS SCRIPT WILL INVESTIGATE BEAT FREQUENCIES IN A LOMB-SCARGLE PERIODOGRAM
def sinewave(xvals, amplitude, angfreq, phase):
	return amplitude * np.sin(angfreq*xvals + phase)


xvals = np.arange(0,51,1) #### epochs
#xvals = np.sort(np.random.choice(xvals, size=int(0.8*len(xvals))))
deltax = xvals[1] - xvals[0]


use_errors = input("Use errors? y/n: ")
if use_errors == 'y':
	dys = np.linspace(0.1, 0.1, len(xvals))
else:
	dys = np.linspace(0,0,len(xvals))

period1 = 0.1 #### fraction of an epoch.
freq1 = 1/period1
period2 = 0.25 #### fraction of an epoch
freq2 = 1/period2 

beatfreq = np.abs(freq1 - freq2)
sine1 = sinewave(xvals=xvals, amplitude=1, angfreq=2*np.pi*freq1, phase=np.pi)
sine2 = sinewave(xvals=xvals, amplitude=1, angfreq=2*np.pi*freq2, phase=np.pi)
sine3 = sine1+sine2 

if use_errors == 'y':
	sine1 = np.random.normal(loc=sine1, scale=dys)
	sine2 = np.random.normal(loc=sine2, scale=dys)
	sine3 = sine1 + sine2

plt.plot(xvals, sine1, c='r', alpha=0.7, linestyle='--', label='sine1')
plt.plot(xvals, sine2, c='b', alpha=0.7, linestyle='--', label='sine2')
plt.plot(xvals, sine3, c='k', alpha=0.7, linewidth=2, label='sine1+sine2')
plt.legend()
plt.show()


#### make a periodogram of all three

period_range = np.logspace(np.log10(2),np.log10(100),1000)
freq_range = 1/period_range
LS1 = LombScargle(xvals, sine1, dys)
LS1_powers = LS1.power(freq_range)
LS2 = LombScargle(xvals, sine2, dys)
LS2_powers = LS2.power(freq_range)
LS3 = LombScargle(xvals, sine3, dys)
LS3_powers = LS3.power(freq_range)

fig, ax = plt.subplots(3, sharex=True)
ax[0].plot(period_range, LS1_powers, c='r', alpha=0.7)
#ax[0].plot(np.linspace(period1, period1, 100), np.linspace(0,1, 100), c='g', linestyle='--')
ax[0].set_ylabel('LS1')
ax[1].plot(period_range, LS2_powers, c='b', alpha=0.7)
ax[1].set_ylabel("LS2")
#ax[1].plot(np.linspace(period2, period2, 100), np.linspace(0,1, 100), c='g', linestyle='--')
ax[2].plot(period_range, LS3_powers, c='k', alpha=0.7)
ax[2].set_ylabel('LS3')
plt.xscale('log')
#ax[2].plot(np.linspace(1/beatfreq, 1/beatfreq, 100), np.linspace(0,1,100), c='g', linestyle='--')
plt.show()


#### find the highest peak of LS3_powers, do a curve_fit subtraction, then see what comes out!
peak_LS3power_idx = np.argmax(LS3_powers)
best_LS3_period = period_range[peak_LS3power_idx]
best_LS3_freq = 1 / best_LS3_period
#### DO CURVE FIT
def LS3_sinewave(xvals, amplitude, phase):
	return amplitude * np.sin((xvals * 2*np.pi * best_LS3_freq) + phase)

LS3popt, LS3pcov = curve_fit(LS3_sinewave, xvals, sine3, sigma=dys)
#### now subtract off this curve

LS3_minus_bestfit = sine3 - LS3_sinewave(xvals, *LS3popt)

### plot it
plt.plot(xvals, sine1, c='r', alpha=0.7, linestyle='--', label='sine1')
plt.plot(xvals, sine2, c='b', alpha=0.7, linestyle='--', label='sine2')
plt.plot(xvals, sine3, c='k', alpha=0.7, linewidth=2, label='sine3')
#### plot the best fit solution
plt.plot(xvals, LS3_sinewave(xvals, *LS3popt), c='g', linewidth=2, label='best fit')
plt.legend()
plt.show()

### LS3_sinewave matches sine2 very well. LOOK AT THE RESIDUAL.
plt.plot(xvals, sine1, c='r', alpha=0.7, linestyle='--', label='sine1')
plt.plot(xvals, LS3_minus_bestfit, c='g', alpha=0.7, label='LS3 minus best fit')
plt.show()

##### now run a periodogram on sine1, and LS3_minus_bestfit
LS3_minus_bestfit_LS = LombScargle(xvals, LS3_minus_bestfit, dys)
LS3_minus_bestfit_powers = LS3_minus_bestfit_LS.power(freq_range)

#### plot them
fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(period_range, LS1_powers, c='r')
ax[1].plot(period_range, LS3_minus_bestfit_powers, c='g')
plt.xscale('log')
plt.show()


secondit_peak_power_idx = np.argmax(LS3_minus_bestfit_powers)
secondit_best_period = period_range[secondit_peak_power_idx]
secondit_best_freq = 1/secondit_best_period

#### now subtract off this second signal, and see if you've got anything left
def sinewave_2ndit(xvals, amplitude, phase):
	return amplitude * np.sin((2*np.pi*secondit_best_freq * xvals) + phase)

secondit_popt, secondit_pcov = curve_fit(sinewave_2ndit, xvals, LS3_minus_bestfit, sigma=dys)
secondit_subtracted_curve = LS3_minus_bestfit - sinewave_2ndit(xvals, *secondit_popt)

##### nothing left(?) curve
plt.plot(xvals, secondit_subtracted_curve)
plt.show()

#### see if there's any power left in the periodogram
last_LS = LombScargle(xvals, secondit_subtracted_curve, dys)
last_LSpowers = last_LS.power(freq_range)
last_FAP = last_LS.false_alarm_probability(last_LSpowers.max())
plt.plot(period_range, last_LSpowers)
plt.xscale('log')
plt.title("FAP = "+str(last_FAP))
plt.show()