import numpy as np
from scipy.interpolate import interp1d
import h5py
import matplotlib.pyplot as plt
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot


def Hann(n, N):
	y = 0.5*(1-np.cos(2*np.pi*n/(N-1)))
	return y


def calcpsd(x, y, window):
	"""Calculate the PSD from the signal time series averaged from the PSDs of 1 second chunks
	Use given window to smooth each fft. Return PSD as an interpolated function"""
	dt = x[1] - x[0]
	neach = int(1/dt)
	nfft = int(2 * np.floor(len(y)/neach) - 1)
	shift = int(np.ceil(neach/2))
	freq = np.fft.rfftfreq(neach, dt)
	asds = np.zeros((nfft, len(freq)), dtype='complex')
	psds = np.zeros((nfft, len(freq)), dtype='complex')
	for i in range(0, nfft):
		windowed = y[i*shift:(i+2)*shift]*window(np.arange(neach), neach)
		asds[i, :] = np.fft.rfft(windowed)
		psds[i, :] = asds[i, :]*np.conj(asds[i, :])
	psd = np.mean(psds, axis=0)
	psdi = interp1d(freq, psd, axis=0)
	return psdi


def whiten(x, y, interp_psd):
	"""use psd to whiten given data"""
	dx = x[1] - x[0]
	num = np.size(x)
	freqs = np.fft.rfftfreq(num, dx)
	yft = np.fft.rfft(y)
	white_yft = yft/(np.sqrt(interp_psd(freqs)))
	white_y = np.fft.irfft(white_yft, n=num)
	return white_y


def bandpass(strain, dt, freq1, freq2):
	num = np.size(strain)
	freqs = np.fft.rfftfreq(num, dt)
	filt = np.zeros((num/2+1), dtype='int')
	for i in range(0, num/2):
		if (freqs[i]>freq1) and (freqs[i]<freq2):
			filt[i] = 1
	sfft = np.fft.rfft(strain)
	sfftbp = sfft*filt
	sbp = np.fft.irfft(sfftbp, n=num)
	return sbp


def qtransform(x, y, qval, window, yeval):
	q = np.zeros((len(x), len(yeval)), dtype='complex')
	dx = x[1] - x[0]
	for i in range(0, len(yeval)):
		fi = yeval[i]
		windowxp = qval/fi
		windown = int(windowxp/dx)
		windowy = window(np.arange(windown), windown)
		windowy *= 1/np.sum(windowy)
		windowx = np.arange(windown)*dx
		windowyf = windowy * (np.cos(2*np.pi*fi*windowx) - 1j*np.sin(2*np.pi*fi*windowx))
		qpre = np.correlate(y, windowyf, mode='full')
		cut = int(np.floor(windown/2))
		q[:, i] = qpre[cut-1+windown%2:-cut]*np.conj(qpre[cut-1+windown%2:-cut])
	return np.real(q)


gps = event_gps('GW170817')
segment = (int(gps) - 60, int(gps) + 10)
hdata = TimeSeries.fetch_open_data('H1', *segment, tag='CLN', verbose=True, cache=True)
"""
hq = hdata.q_transform(frange=(30, 500), qrange=(100, 100))
plot = hq.plot()
plot.show()
"""

strain = np.array(hdata.real)
ts = np.array(hdata.times - hdata.t0)
fs = int(np.array(hdata.sample_rate))
print 'time =', (ts[-1] - ts[0]), ' nsamples =', hdata.size, ' sample rate =', fs

"""
psdi = calcpsd(ts, strain, Hann)
strainw = whiten(ts, strain, psdi)
strainbp = bandpass(strainw, ts[1]-ts[0], 10, 600)
"""

f1, f2 = 30, 500
feval = np.geomspace(f1, f2, endpoint=True, num=500)
#feval = np.linspace(f1, f2, endpoint=True, num=100)
q = qtransform(ts, strain, 50, Hann, feval)
logq = np.log(q**0.5)
plt.imshow(np.transpose(logq[fs*5:-fs*5]), cmap=plt.get_cmap('winter'), origin='lower', aspect='auto')
plt.colorbar()
plt.show()
