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


def spectrogram(x, y, windowtype, dxspec, ymin, ymax):
	"""create a 2D spectrogram from time-series data
	define type of window to be used. dxspec defines f and x resolution"""
	dx = x[1] - x[0]
	neach = int(np.floor(dxspec*2./dx))
	xstart = int(np.floor(neach/2.))
	xend = len(x) - xstart - 1
	jump = xstart + 1
	nfft = (xend - xstart)/jump
	xspec = np.linspace(x[xstart], x[xend], num=nfft)
	yspec = np.fft.rfftfreq(neach, dx)
	z = np.zeros((nfft, len(yspec)), dtype='complex')
	window = windowtype(np.arange(neach), neach)
	print 'nfft =', nfft, ' jump =', jump
	for i in range(0, nfft):
		asd = np.fft.rfft(y[i*jump:i*jump+neach]*window)
		z[i, :] = asd*np.conj(asd)
	i1 = np.argmax(yspec > ymin)
	i2 = np.argmax(yspec > ymax)
	return xspec, yspec[i1:i2], np.real(z[:, i1:i2])


gps = event_gps('GW170817')
segment = (int(gps) - 60, int(gps) + 10)
hdata = TimeSeries.fetch_open_data('H1', *segment, tag='CLN', verbose=True, cache=True)
"""
specgram = hdata.spectrogram2(fftlength=0.5, window='hann')
plot = specgram.plot(norm='log')
ax = plot.gca()
ax.set_ylim(30, 500)
plot.show()
"""

strain = np.array(hdata.real)
ts = np.array(hdata.times - hdata.t0)
fs = int(np.array(hdata.sample_rate))
print 'time =', (ts[-1] - ts[0]), ' nsamples =', hdata.size, ' sample rate =', fs

"""
psdi = calcpsd(ts, strain, Hann)
strainw = whiten(ts, strain, psdi)
"""

f1, f2 = 30, 500
t, f, spec = spectrogram(ts, strain, Hann, 0.5, f1, f2)
plt.imshow(np.transpose(np.log(spec)), cmap=plt.get_cmap('winter'), origin='lower', aspect='auto', extent=[t[0],t[-1],f[0],f[-1]])
plt.show()
