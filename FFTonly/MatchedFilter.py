import numpy as np
from scipy.interpolate import interp1d
import h5py
import matplotlib.pyplot as plt


def SGsignal(length, freq, freqm, max_amp):
	"""Define function that takes the required length in samples, the required freq in samples, the frquency multiplier and the max amplitude
	Return a sine function with frequency increasing with time to freq*freqm, starting at period, multiplied by a gaussian with SD of length/6"""
	sd = length/6
	x = np.arange(length) - (length-length%2)/2.
	k = 2*np.pi*freq * np.linspace(1, freqm, num=length)
	y = max_amp * np.exp(-(x**2)/(2*(sd**2))) * np.sin(k*x)
	return x, y


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
	psdtime = np.fft.irfft(psd)[length//2-int(2//dt):length//2+int(2//dt)]
	psd = np.fft.rfft(psdtime)
	psdfreq = np.fft.rfftfreq(len(psdtime), dt)
	psdi = interp1d(psdfreq, psd, axis=0)
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


def bandpass(x, y, f1, f2):
	"""Apply a bandpass filter to timeseries data"""
	xft = np.fft.rfftfreq(len(x), x[1] - x[0])
	yft = np.fft.rfft(y)
	filt = np.zeros(len(xft), dtype='int')
	for i in range(0, len(xft)):
		if (xft[i]>f1) and (xft[i]<f2):
			filt[i] = 1
	yft = yft*filt
	y = np.fft.irfft(yft)
	return y


def inner(x1, y1, x2, y2, psdi):
	"""Perform the inner product of two signals given in the time domain
	return the resulting time-series. Assuming the resolutions are equal"""
	dx = x1[1] - x1[0]
	if len(y1) < len(y2):
		xft = np.fft.rfftfreq(len(x2), dx)
		y2ft = np.fft.rfft(y2)
		y1temp = np.zeros((len(x2)), dtype='float')
		y1temp[:len(y1)] += y1
		y1ft = np.fft.rfft(y1temp)
	elif len(y2) < len(y1):
		xft = np.fft.rfftfreq(len(x1), dx)
		y1ft = np.fft.rfft(y1)
		y2temp = np.zeros((len(x1)), dtype='float')
		y2temp[:len(y2)] += y2
		y2ft = np.fft.rfft(y2temp)
	else:
		xft = np.fft.rfftfreq(len(x1), dx)
		y1ft = np.fft.rfft(y1)
		y2ft = np.fft.rfft(y2)
	psd = psdi(xft)
	innerft = y1ft*np.conj(y2ft)/psd
	yinner = np.fft.irfft(innerft)
	dt = 1/(len(yinner)*(xft[1] - xft[0]))
	xinner = np.linspace(0, len(yinner)*dt, num=len(yinner))
	return xinner, yinner


def chi_squared(xh, yh, xs, ys, psdi, nbins):
	templates = np.zeros((len(xh), nbins), dtype='float')
	power = sum(yh**2)
	used = 0
	pbins = np.zeros((nbins), dtype='float')
	for i in range(0, nbins-1):
		pbin = 0
		count = 0
		while pbin < power/nbins:
			templates[used+count, i] = yh[used+count]
			pbin += yh[used+count]**2
			count += 1
		pbins[i] = pbin
		used += count
	templates[used:, nbins-1] = yh[used:]
	pbins[-1] = sum(templates[:, -1]**2)
	"""
	for i in range(0, nbins):
		plt.subplot(nbins, 1, i+1)
		plt.plot(xh, templates[:, i])
	plt.show()
	"""
	x, snr = inner(xh, yh, xs, ys, psdi)
	x, snr = x[len(xh)*10:-len(xh)*10], np.abs(snr[len(xh)*10:-len(xh)*10])
	psnr = np.zeros((len(snr), nbins), dtype='float')
	for i in range(0, nbins):
		_, psnr0 = inner(xh, templates[:, i], xs, ys, psdi)
		psnr[:, i] = np.abs(psnr0[len(xh)*10:-len(xh)*10])
	chi_each = np.zeros((len(snr), nbins), dtype='float')
	for i in range(0, nbins):
		chi_each[:, i] = (psnr[:, i] - pbins[i])**2
	"""
	for i in range(0, nbins):
		plt.plot(x, psnr[:, i])
	plt.show()
	for i in range(0, nbins):
		plt.plot(x, chi_each[:, i])
	plt.show()
	"""
	chi = np.sum(chi_each, axis=1)
	rsnr = np.zeros((len(snr)), dtype='float')
	for i in range(0, len(snr)):
		if chi[i] > 1:
			rsnr[i] = snr[i]/(((1 + chi[i]**3)/2.)**(1/6.))
		else:
			rsnr[i] = snr[i]
	return x, snr, chi, rsnr


#Read a peice of strain data and make a time vector
filename = '../../Data/H-H1_LOSC_4_V1-1126076416-4096.hdf5'
f = h5py.File(filename, 'r')
strain = f['strain/Strain'][()]
dt = f['strain/Strain'].attrs['Xspacing']
tstart = f['meta/GPSstart'][()]
tend = f['meta/Duration'][()] + tstart
ts = np.arange(0, tend-tstart, dt)
fs = int(1/dt)
print('time = ', (tend - tstart), ' namples = ', len(strain), ' sample rate = ', fs)

"""
Create a fake signal to test and plot the amplitude of the time series and fourier transform
length, freq, freqm, maxa = int(0.1*fs), 100./fs, 3, 1e-21
x, y = SGsignal(length, freq, freqm, maxa)
xft = np.fft.rfftfreq(length, dt)
yft = np.fft.rfft(y)
ypsd = yft*np.conj(yft)

plt.subplot(2, 1, 1)
plt.plot(x*dt, y)
plt.subplot(2, 1, 2)
plt.plot(xft, yft)
plt.axvline(freq/dt, c='blue')
plt.axvline(freq*freqm/dt, c='red')
plt.show()
"""

#Place signal with known parameters into the LIGO dataat unknown time
lengths = 1 #length of signal in seconds
freqs = 40. #freq in seconds^-1
length, freq, freqm, maxa = int(lengths*fs), freqs/fs, 5, 1e-21
xh, yh = SGsignal(length, freq, freqm, maxa)

eventi = int( np.random.rand(1) * (len(strain) - length) )
signal = np.zeros((len(strain)), dtype='float')
signal[eventi:eventi+int(length)] += yh
data = strain + signal
print('Event time = ', eventi*dt)

psdi = calcpsd(ts, data, Hann)

bp = bandpass(ts, data, 30, 600)

swhite = whiten(ts, bp, psdi)

t, snr, chi, rsnr = chi_squared(xh*dt, yh, ts, swhite, psdi, 8)

print('Max SNR at t = ', ts[np.argmax(snr)])
print('Min CHI at t = ', ts[np.argmin(chi)])
print('Max RSNR at t = ', ts[np.argmin(rsnr)])

plt.subplot(4, 1, 1)
plt.title('whitened data')
plt.plot(ts, swhite)
#plt.xlim([eventi*dt-0.5*lengths, eventi*dt+1.5*lengths])
#plt.ylim([-0.1, 0.1])
plt.subplot(4, 1, 2)
plt.title('SNR')
plt.plot(t, snr)
#plt.xlim([eventi*dt-0.5*lengths, eventi*dt+1.5*lengths])
#plt.ylim([-1e18, 1e18])
plt.subplot(4, 1, 3)
plt.title('Chi_squared')
plt.plot(t, chi)
#plt.xlim([eventi*dt-0.5*lengths, eventi*dt+1.5*lengths])
#plt.ylim([0, 1e36])
plt.subplot(4, 1, 4)
plt.title('RSNR')
plt.plot(t, rsnr)
#plt.xlim([eventi*dt-0.5*lengths, eventi*dt+1.5*lengths])
#plt.ylim([0, 1e36])
plt.show()
