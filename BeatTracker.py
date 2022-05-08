# libraries
import os
import numpy as np
import librosa
import librosa.display
# from matplotlib import pyplot as plt
import mir_eval
from scipy.ndimage import uniform_filter1d as meanfilt
from scipy.ndimage import maximum_filter1d as maxfilt
from scipy.ndimage import median_filter as medfilt
from scipy.ndimage import gaussian_filter as gaussfilt
from scipy.interpolate import interp1d
import scipy.signal as signal
import scipy.io.wavfile as wavfile
from scipy import stats
import IPython.display as ipd
import math


# ----- Spectrogram ----- #

# get a spectrogram from an audio signal
def spectro(inFile, sr=22050, hopTime=0.023, windowTime=0.046, window='hamming', 
            mel=False):
    # get audio data
    snd, rate = librosa.load(inFile, sr=sr)
    # number of samples per hop
    hop = int(2 ** np.ceil(np.log2(rate * hopTime)))
    # round window length up to the next power of 2
    wlen = int(2 ** np.ceil(np.log2(rate * windowTime)))
    
    # get fft for each time window
    if mel == False:
        spectrogram = librosa.stft(snd, n_fft=wlen, hop_length=hop, win_length=wlen,
                                   window=window)
    else:
        spectrogram = librosa.feature.melspectrogram(y=snd, sr=rate, n_fft=wlen, 
                                                     hop_length=hop, window=window, power=1.0)
    # hop rate (hops per second)
    hr = rate / hop
    return spectrogram, hr


# ----- Spectral Flux Novelty Curve ----- #

# moving filter: subtract local mean and rectify
def local_av(arr, filtWin, fType='mean', avWeight=1):
    # mean filter
    if fType == 'mean':
        L = len(arr)
        mean = np.zeros(L)
        for m in range(L):
            a = max(m - filtWin, 0)
            b = min(m + filtWin + 1, L)
            mean[m] = (1 / (2 * filtWin + 1)) * np.sum(arr[a:b])
            
    # median filter
    elif fType == 'median':
        mean = medfilt(arr, filtWin)
        
    # subtract local mean and half-wave rectify
    mean = mean * avWeight
    arr = np.multiply(np.greater(arr, mean), np.subtract(arr, mean))
    return arr


# calculate spectral flux novelty curve from spectrogram
def fluxCurve(spectr, hr, meanFiltTime=0.1, log_mag=True, 
              bands=1, diff=1, bandWeights=None, lib=False,
              window='hamming', avType='mean', avWeight=1, 
              logScalar=1000.0, maxFilt=False, logBands=False):
    """
    Calculate spectral flux onset detection with input parameters:
         spectr: spectrogram
         hr: spectrogram hop rate
         meanFiltTime: time (s) of the smoothing mean filter applied to the final 
                  novelty curve
         log_mag: take a log of the magnitudes
         bands: novelty curves are calculated in bands over the frequency domain - this
                determines the number of bands         
         diff: number of time windows back for difference
         bandWeights: relative weighting of each band from lowest to highest
                  Note: must be the same length as 'bands'
         lib: Use the librosa flux curve (for comparison)
         window: window type
         avType: average type
         avWeight: Scaling factor for the moving average filter
         logScalar: scale factor for the logarithmic compression
         maxFilt: apply a max filter to the flux values along the frequency axis
         logBands: split the frequency bands by a logarithmic scale
    Output: vector of onset densities per hop
         
    """
    # magnitude spectrogram
    spectr = np.abs(spectr)
    # account for logarithmic perception of sound intensity
    if log_mag:
        C = logScalar  # scaling constant
        spectr = np.log(1 + C * spectr)
    
    # maximum filter for superflux
    if maxFilt:
        for t in range(spectr.shape[1]):
            spectr[:, t] = maxfilt(spectr[:, t], size=3)
        
    # calculate flux values
    flux = np.diff(spectr, n=diff)
    flux = np.concatenate([flux, np.zeros((spectr.shape[0], diff))], axis=1)
    
    # librosa onset_strength for comparison
    if lib == True:
        nc = librosa.onset.onset_strength(S=spectr, lag=diff)
        return nc
    
    # half-wave rectify
    flux[flux < 0] = 0
    
    if bands == 1:
        flux = np.sum(flux, axis=0)
    else:
        if not logBands:
            # split into linearly spaced non-overlapping frequency bands
            freqInd = np.linspace(0, spectr.shape[0], num=bands+1, 
                                  endpoint=True)
            freqInd = freqInd.astype(int)
        else:
            # split into logarithmically-spaced frequency bands
            base = np.power(spectr.shape[0], 1.0 / bands)
            freqInd = np.logspace(1, bands, num=bands, endpoint=True, 
                                       base=base, dtype=int)
            freqInd = np.concatenate([[0], freqInd])
        
        # calculate spectral flux for each band and sum
        fluxes = np.zeros(flux.shape[1])
        # initialise bandWeights
        if not bandWeights:
            bandWeights = np.ones(bands)
        for i in range(bands):
            band = flux[freqInd[i]:freqInd[i + 1], :]
            band = np.sum(band, axis=0)
            fluxes = fluxes + band * bandWeights[i]
        flux = fluxes

    # subtract local mean and rectify
    ker_len = int(2 * ((meanFiltTime * hr) // 2) + 1)
    flux = local_av(flux, ker_len, fType=avType)
    
    return flux / np.max(flux)


# Smooth curve and resample
def smoothResample(nc, hr, hr_out=100, norm=True, sigma=1, maxTime_sec=None):
    # apply a smoothing filter (sigma: standard deviation)
    if sigma:
        nc = gaussfilt(nc, sigma=sigma)
        
    # initialise output
    T_in = np.arange(nc.shape[0]) / hr
    maxTime = T_in[-1]
    if maxTime_sec is None:
        maxTime_sec = maxTime
    N_out = int(np.ceil(maxTime_sec*hr_out))
    T_out = np.arange(N_out) / hr_out
    if T_out[-1] > maxTime:
        nc = np.append(nc, [0])
        T_in = np.append(T_in, [T_out[-1]])
        
    # apply interpolation
    x_out = interp1d(T_in, nc, kind='linear')(T_out)
    
    # normalise
    if norm:
        x_max = max(x_out)
        if x_max > 0:
            x_out = x_out / max(x_out)
    return x_out, hr_out


# locations of onsets
def get_onsets(nc, filtWin=3, thresh=0):
    # subtract local median
    medFiltnc = np.subtract(nc, medfilt(nc, size=filtWin))
    
    # find local maximum values
    maxFiltnc = maxfilt(medFiltnc, filtWin, mode='nearest')
    
    # set threshold to maximum of local max or the 'thresh'
    threshold = [max(i, thresh) for i in maxFiltnc]
    
    # onsets are where the median filtered curve is above the threshold
    onsets = np.greater_equal(medFiltnc, threshold)
    
    # get list of indices of onsets
    peakIndices = np.nonzero(onsets)
    
    return peakIndices, onsets


# ----- Tempogram ----- #

# calculate tempogram from novelty curve
def tempogram(noveltyCurve, hopRate, winTime=10.0, minBPM=50, maxBPM=250, 
              BPMres=1.0, tHopTime=0.1, tType='Fourier', avType='mean'):
    """
    Calculate tempogram for an onset function:
        hopRate: number of hop samples per second (output from noveltyCurve())
        winTime: time for a tempogram window 
        minBPM: minimum BPM value (minimum of frequency axis of tempogram)
        maxBPM: maximum BPM value
        BPM res: gap between two consecutive BPM values
        tHopTime: time between hops
        tType: 'Fourier' for Fourier tempogram, 'auto' for autocorrelation
        avType: average type
    output: matrix representing tempogram
            Frequency values are [minBPM:maxBPM] BPM
    """
    # remove any unnecessary dimensions from the NC
    noveltyCurve = np.squeeze(noveltyCurve)
    
    # number of novelty curve samples per tempogram hop
    tHop = int(np.round(hopRate * tHopTime))
    # window length for novelty curve samples
    tWlen = 2 * int(np.round(hopRate * winTime) / 2)
    # number of windows
    num_win = int(len(noveltyCurve) / tHop)  + 1
    # window function
    win = np.hanning(tWlen)
    # normalise window
    win = win / np.sum(win)
    
    # add zero-padding to novelty curve
    noveltyCurve = np.squeeze(noveltyCurve)
    nc_pad = np.concatenate([np.zeros(int(tWlen / 2)), noveltyCurve, 
                                   np.zeros(int(tWlen / 2))])
    
    
    # get phasor frequencies from BPM values
    freq = np.arange(minBPM, maxBPM + 1, BPMres)
    # change BPM to BPS
    freq = np.divide(freq, 60.0)
    t_pad = np.arange(len(nc_pad)) / hopRate

    # Fourier tempogram
    if tType == 'Fourier':
        # initialise tempogram
        tempo = np.zeros((len(freq), num_win)).astype(complex)

        # calculate Fourier Coefficients
        for k, f in enumerate(freq):
            phasor = np.exp(-2j * np.pi * f * t_pad)
            x = np.multiply(nc_pad, phasor)
            for i in range(num_win):
                t0 = i * tHop
                t1 = t0 + tWlen
                tempo[k, i] = np.sum(win * x[t0: t1])
    
    # autocorrelation tempogram
    elif tType == 'auto':
        # initialise tempogram
        tempo = np.zeros((len(freq), num_win))
        nc_pad = np.concatenate([noveltyCurve, np.zeros(2 * tWlen)])
        # calculate correlation coefficients
        for i in range(num_win):
            x = nc_pad[int(i * tHop): int(i * tHop + tWlen)]

            for n in range(len(freq)):
                # number of samples for delay
                delay = int(hopRate / freq[n])
                # delayed signal
                x_delay = nc_pad[int(i * tHop) + delay: 
                                       int(i * tHop + tWlen) + delay]
                # calculate correlation
                coeff = np.multiply(x, x_delay)
                coeff = np.multiply(coeff, win)
                coeff = np.sum(coeff)
                # scale
                tempo[n, i] = coeff / (tWlen - delay)
                
    # rate of time bins on the tempogram per second
    tHopRate = hopRate / tHop
    
    return tempo, tHopRate


# combine Fourier and autocorrelation tempograms together (weighted sum)
def combinedMagTemp(fourTemp, autoTemp, autoWeight=0.5):
    """
    Combines two tempograms of the same dimensions
    The values must be real
    """
    # find maximum values
    maxFour = np.max(fourTemp)
    maxAuto = np.max(autoTemp)
    # scale by maximum values
    fourTemp = np.divide(fourTemp, maxFour)
    autoTemp = np.divide(autoTemp, maxAuto)
    
    return fourTemp + autoTemp * autoWeight


# Constrain a tempogram to a range either side of the modal tempo
def constrainTempo(temp, fourTemp, locations, minBPM, maxBPM, BPMres, 
                   factor=0.5):
    # find modal maximum tempo
    modeLoc = stats.mode(locations)[0][0]
    ##print(modeLoc)
    # modal tempo
    modeTempo = minBPM + modeLoc * BPMres
    ##print(modeTempo)
    # constrain to range of +- factor * x
    newMinT = max(int(int((1 - factor) * modeTempo / BPMres) * \
                      BPMres), minBPM)
    newMaxT = min(int(int((1 + factor) * modeTempo / BPMres) * \
                      BPMres), maxBPM)
    # find corresponding tempo locations
    minLoc = int((newMinT - minBPM) / BPMres)
    maxLoc = temp.shape[0] - int((maxBPM - newMaxT) / BPMres)
    return temp[minLoc: maxLoc, :], fourTemp[minLoc: maxLoc + 1, :], \
        newMinT, newMaxT


# ----- PLP curve ----- #

# Calculate the predominant Local Pulse
def PLP(fourTemp, hr, thr, combTemp=None, winTime=9, minBPM=50, maxBPM=250, 
        BPMres=1.0, constrain=False, constrainFactor=0.5):
    """
    Computes the Predominant Local Pulse Curve
    Input:
        fourTemp: a Fourier Tempogram (must be in complex form)
        hr: Hop rate of the novelty curve
        thr: number of time bins per second for the tempogram(s)
        combTemp: option to pass in a combined Fourier and autocorrelation 
            tempogram.  This will have real values. This matrix must be 
            the same size as fourTemp
        winTime: window time (s)
        constrain: optiona to use the constrainTempo() function
        constrainFactor: factor by which to set the tempo range either side of 
            the modal value
    Output:
        array of PLP values, 1 for each hop time
    """
    # rate of the PLP curve relative to the tempogram hop rate
    rate = int(hr / thr)
    
    # window length (odd)
    winLen = 2 * int(np.round(hr * winTime) / 2) + 1
    
    # get matrix for magnitude values
    magMat = np.abs(fourTemp) if not np.any(combTemp) else combTemp
    # find locations for maximum value at each time point
    locations = np.argmax(magMat, axis=0)
    
    # constrain the tempo range relative to the modal value
    if constrain:
        magMat, fourTemp, minBPM, maxBPM = constrainTempo(magMat, fourTemp, 
                                                          locations, minBPM, 
                                                          maxBPM, BPMres, 
                                                          factor=constrainFactor)
        # find new max locations
        locations = np.argmax(magMat, axis=0)
        ##print(magMat.shape)
    
    pnts = range(len(locations))
    # tempo values at each location
    tempVals = np.arange(minBPM, maxBPM + 1, BPMres)

    # tempi = [tempVals[locations[i]] for i in pnts]
    tempi = [tempVals[locations[i]] for i in pnts]
    
    # array of magnitudes
    mags = [np.abs(fourTemp[locations[i], i]) for i in pnts]
    
    # array of corresponding phases
    reals = [np.real(fourTemp[locations[i], i]) for i in pnts]
    
    # phases = [1 / (2 * np.pi) * np.arccos(reals[i] / mags[i]) for i in pnts]
    phases = [1 / (2 * np.pi) * np.arccos(reals[i] / mags[i]) for i in pnts]
    
    # zero-pad length
    numZP = winLen // 2
    
    # initialise kernels
    kernels = np.zeros(len(locations) * rate + 2 * numZP)
    
    # window function
    win = np.hanning(winLen)
    # normalise window
    win = win / np.sum(win) * rate
    
    # calculate the sinusoidal periodicity kernel for each time location
    for t in pnts:
        # time points
        n = np.arange(int(t * thr), int(t * thr + winLen))
        # kernel
        ker = np.cos(2 * np.pi * ((n / hr) * tempi[t] / 60 - phases[t]))
        ker = np.multiply(ker, win)
        # overlap sum
        kernels[n] = kernels[n] + ker
        
    # remove padding
    kernels = kernels[numZP:-numZP]
    # half-wave rectify
    kernels = np.where(kernels > 0, kernels, 0)
    
    return kernels, hr


# ----- Salience Features ----- #

# utility function to remove outliers from a signal
def remove_outliers(arr, max_dev=3.5):
    # median value
    med = np.median(arr)
    # squared deviation from median
    diff = (arr - med) ** 2
    # take the root
    diff = np.sqrt(diff)
    
    # median deviation
    mad = np.median(diff)
    # set outlier threshold
    if mad == 0:
        score = 0
    else:
        score = 0.6745 * diff / mad
    # remove outliers
    arr = np.where(score > max_dev, med, arr)
    return arr


# calculate salience features

# energy (not used)
def rms(spectr, filtWin=25, avType='mean'):
    fl = 2 * spectr.shape[0] - 1
    amp = librosa.feature.rms(S=np.abs(spectr), frame_length=fl)
    # remove local mean and rectify
    amp = np.squeeze(amp)
    amp = local_av(amp, filtWin, avType)
    return amp / np.amax(amp)


# get a chromagram
def chroma(path, sr, hopTime=0.023, windowTime=0.026, threshold=0, 
           bins=12, cType='cqt', norm=None, window='hann'):
    # number of samples per hop
    snd, rate = librosa.load(path, sr=sr)
    hop = round(rate * hopTime)
    hopCQT = None
    # round window length up to the next power of 2
    wlen = int(2 ** np.ceil(np.log2(rate * windowTime)))
    # calculate chromagram (3 different options)
    if cType == 'cqt':
        hop = int(64 ** np.ceil(math.log(hop, 64)))
        cgm = librosa.feature.chroma_cqt(y=snd, sr=rate, hop_length=hop, 
                                         threshold=threshold, n_chroma=bins)
    elif cType == 'stft':
        cgm = librosa.feature.chroma_stft(y=snd, sr=rate, norm=np.inf, 
                                          n_fft=wlen, hop_length=hop, 
                                          window=window, n_chroma=bins)
    elif cType == 'cens':
        cgm = librosa.feature.chroma_cens(y=snd, sr=rate, hop_length=hop,
                                          bins_per_octave=bins, norm=2)
    
    hr = rate / hop
    return cgm, hr


# chromatic deviation from the mean chroma position
def harmonicDev(chromagram, hr, pow=1, local=False, localLength=100):
    # find locations of maximum values
    maxLoc = np.argmax(chromagram, axis=0)
    if not local:
        # modal position
        modalLoc = stats.mode(maxLoc)[0][0]
    else:
        modalLoc = np.zeros(chromagram.shape[1])
        # find locations of local modal maximum values
        for i in range(chromagram.shape[1]):
            left = max(0, i - localLength // 2)
            right = min(chromagram.shape[1], i + localLength // 2)
            modalLoc[i] = stats.mode(maxLoc[left: right])[0][0]
    # find deviation from [local] modal maximum
    dev = np.abs(maxLoc - modalLoc)
    # rescale and take power
    dev = np.power(np.divide(1, dev + 1), pow)
    return dev / np.amax(dev)


# spectral contrast
def density(spectr, filtWin=3, outlierThresh=10.5, av=True, 
            avWin=100, avType='mean'):
    spectr = np.abs(spectr)
    dens = librosa.feature.spectral_centroid(S=spectr)
    dens = np.divide(1, dens + 1)

    # median filter to remove clicks
    dens = remove_outliers(dens, max_dev=outlierThresh)

    # filter to bring out onsets
    dens = np.squeeze(dens)
    if av:
        dens = local_av(dens, avWin, avType)
    return dens / np.amax(dens)


# spectral flatness (noisiness)
def noisy(spectr, power=2, outlierThresh=10.5):
    spectr = np.abs(spectr)
    noise = librosa.feature.spectral_flatness(S=spectr, power=power)
    # remove outliers
    noise = np.squeeze(noise)
    noise = remove_outliers(noise, max_dev=outlierThresh)
    return noise / np.max(noise)


# Number of samples between  (not used)
def IOI(oInd, ncLen, startOnly=True):
    # number of samples until next onset
    oInd = np.squeeze(oInd)
    iois = np.squeeze(np.diff(oInd))
    ioi = np.zeros(ncLen)
    for n in range(len(iois)):
        ioi[oInd[n]] = min(iois[n], 100)
    m = np.amax(ioi)
    # normalise
    if m > 0:
        ioi = ioi / m
    return ioi


# combine salience features
def salience(dens, harm, noise, weights=[1, 0.3, 0.6], 
             filtWin=25, fType='mean'):
    nc = np.zeros(len(dens))
    nc = nc + dens * weights[0]
    nc = nc + harm * weights[1]
    nc = nc + noise * weights[2]
    nc = local_av(nc, filtWin=filtWin, fType=fType)       
    return nc / np.amax(nc)


# ----- Get beat predictions ----- #

# get PLP maximum values
def beats_from_PLP(plp, hopRate, filtTime=0.15, thresh=0.3):
    # note that PLP is normalised to a maximum value of 1
    
    # calculate filter length
    wlen = int(np.round(hopRate * filtTime))
    # local maximum filter
    maxf = maxfilt(plp, size=wlen)
    # take local maximum
    threshold = [max(i, thresh) for i in maxf]
    
    # plp curve with only peak values
    plp = np.greater_equal(plp, threshold)
    
    # indices of peak positions
    peakIndices = np.nonzero(plp)
    
    return peakIndices, plp


# get beat times
def beatTimes(predIndices, hopRate):
    # number of seconds per index
    hopTime = 1 / hopRate
    # predicted beat times
    predIndices = np.multiply(predIndices, hopTime)
    return np.squeeze(predIndices)


# ----- Post-processing ----- #

# snap predicted times to note onsets
def snapToOnsets(beatPreds, onsetTimes):
    for i, beat in enumerate(beatPreds):
        idx = np.searchsorted(onsetTimes, beat, side='left')
        if idx > 0 and (idx == len(onsetTimes) or 
                        math.fabs(beat - onsetTimes[idx-1]) < 
                        math.fabs(beat - onsetTimes[idx])):
            beatPreds[i] = onsetTimes[idx - 1]
        else:
            beatPreds[i] = onsetTimes[idx]
    return np.unique(beatPreds)


# ----- Put it all together ----- #

# function to allow parameter tuning - returns beat predictions
def getPLP(path, sr=22050, winTime=9.0, minBPM=70, maxBPM=230, 
           BPMres=1, bands=1, tType='Fourier', diff=1, combWeight=0.5, 
           PLPwinTime=9.0, displayTime=None, beatDir=None, mel=False, 
           ncWindow='hanning', lib=False, bandWeights=None, 
           avWeight=1, ncWeight=1, salWeight=0, log_mag=False, 
           hopTime=0.023, maxFilt=False, logScalar=1000.0,
           salWeights=[1, 0.3, 0.6], beatFiltTime=0.2,
           beatThresh=0.3, snap=False, 
           onsetThresh=0.001, logBands=False, constrain=False, 
           constrainFactor=0.5, harmLocal=True):
    
    # spectrogram
    spectr, shr = spectro(path, sr=sr, mel=mel, hopTime=hopTime)
    
    # spectral flux novelty curve
    nc = fluxCurve(spectr, shr, diff=diff, window=ncWindow, 
                   lib=lib, bandWeights=bandWeights, bands=bands, 
                   avWeight=avWeight, log_mag=log_mag, maxFilt=maxFilt, 
                   logScalar=logScalar, logBands=logBands)
    
    # Smooth and resample to 100 Hz
    nc, hr = smoothResample(nc, shr)
    
    # onset predictions
    PI, onsets = get_onsets(nc, thresh=onsetThresh)
    
    # get salience
    sal = 0
    if salWeight != 0:
        
        # chromogram
        cg, cghr = chroma(path, sr, hopTime=0.023, windowTime=0.026, threshold=0.1, 
                        bins=12, cType='cqt', norm=None, window='hann')
        
        # salience features
        sal1 = density(spectr, avType='mean')
        sal1, hr = smoothResample(sal1, shr, hr_out=100)
        #sal2 = rms(spectr) (not used)
        sal3 = harmonicDev(cg, cghr, pow=2, local=harmLocal)
        sal3, _ = smoothResample(sal3, cghr, hr_out=100)
        if len(nc) > len(sal3):
            sal3 = np.concatenate([sal3, np.zeros(len(nc) - len(sal3))])
        elif len(sal3) > len(nc):
            sal3 = sal3[:len(nc)]
        sal4 = noisy(spectr)
        sal4, _ = smoothResample(sal4, shr, hr_out=100)
        #sal5 = IOI(onsets) (not used)
        sal = salience(sal1, sal3, sal4, weights=salWeights)
    
    nc = ncWeight * nc + sal * salWeight
    
    tempC = None
    # get tempogram
    tempF, thr = tempogram(nc, hr, winTime=winTime, minBPM=minBPM, 
                          maxBPM=maxBPM, BPMres=BPMres, tType='Fourier')
    if tType == 'both': 
        tempA, thr = tempogram(nc, hr, winTime=winTime, minBPM=minBPM, 
                          maxBPM=maxBPM, BPMres=BPMres, tType='auto')
        tempC = combinedMagTemp(np.abs(tempF), tempA, combWeight)
    # get PLP
    plp, plprate = PLP(tempF, hr, thr, tempC, winTime=PLPwinTime, minBPM=minBPM, 
                       maxBPM=maxBPM, BPMres=BPMres, constrain=constrain, 
                       constrainFactor=constrainFactor)
    
    # predicted beat times
    peakIndices, plpBeats = beats_from_PLP(plp, plprate, 
                                           filtTime=beatFiltTime, 
                                           thresh=beatThresh)
    beatPreds = beatTimes(peakIndices, plprate)
    
    # option to snap predictions to note onsets
    if snap:       
        PI = np.squeeze(PI)
        PI = np.divide(PI, hr)
        beatPreds = snapToOnsets(beatPreds, PI)
    
    return beatPreds


# get downbeat predictions
def getDBPLP(path, sr=22050, winTime=9.0, minBPM=15, maxBPM=75, 
           BPMres=1, bands=1, tType='Fourier', diff=1, combWeight=0.5, 
           PLPwinTime=9.0, displayTime=None, beatDir=None, mel=False, 
           ncWindow='hanning', lib=False, bandWeights=None, 
           avWeight=1, ncWeight=0, salWeight=1, log_mag=False, 
           hopTime=0.023, maxFilt=False, logScalar=1000.0,
           salWeights=[0.6, 1, -0.6], beatFiltTime=0.2,
           beatThresh=0.3, snap=False, useOnsets=False, 
           onsetThresh=0.001, logBands=False, constrain=False, 
           constrainFactor=0.5, harmLocal=True):
    # spectrogram
    spectr, shr = spectro(path, sr=sr, mel=mel, hopTime=hopTime)
    # novelty curve - spectral flux
    nc = fluxCurve(spectr, shr, diff=diff, window=ncWindow, 
                   lib=lib, bandWeights=bandWeights, bands=bands, 
                   avWeight=avWeight, log_mag=log_mag, maxFilt=maxFilt, 
                   logScalar=logScalar, logBands=logBands)
    nc, hr = smoothResample(nc, shr)
    
    # onset predictions
    PI, onsets = get_onsets(nc, thresh=onsetThresh)
    
    # get salience
    sal = 0
    if salWeight != 0:
        # chromogram
        cg, cghr = chroma(path, sr, hopTime=0.023, windowTime=0.026, threshold=0.1, 
                        bins=12, cType='cqt', norm=None, window='hann')
        sal1 = density(spectr, avType='mean')
        sal1, hr = smoothResample(sal1, shr, hr_out=100)
        #sal2 = rms(spectr)
        sal3 = harmonicDev(cg, cghr, pow=2, local=harmLocal)
        sal3, _ = smoothResample(sal3, cghr, hr_out=100)
        if len(nc) > len(sal3):
            sal3 = np.concatenate([sal3, np.zeros(len(nc) - len(sal3))])
        elif len(sal3) > len(nc):
            sal3 = sal3[:len(nc)]
        sal4 = noisy(spectr)
        sal4, _ = smoothResample(sal4, shr, hr_out=100)
        #sal5 = IOI(onsets)
        sal = salience(sal1, sal3, sal4, weights=salWeights)
    
    nc = ncWeight * nc + sal * salWeight
    
    tempC = None
    # get tempogram
    tempF, thr = tempogram(nc, hr, winTime=winTime, minBPM=minBPM, 
                          maxBPM=maxBPM, BPMres=BPMres, tType='Fourier')
    if tType == 'both': 
        tempA, thr = tempogram(nc, hr, winTime=winTime, minBPM=minBPM, 
                          maxBPM=maxBPM, BPMres=BPMres, tType='auto')
        tempC = combinedMagTemp(np.abs(tempF), tempA, combWeight)
        
    # get PLP
    plp, plprate = PLP(tempF, hr, thr, tempC, winTime=PLPwinTime, minBPM=minBPM, 
                    maxBPM=maxBPM, BPMres=BPMres, constrain=constrain)
    
    # predicted beat times
    peakIndices, plpBeats = beats_from_PLP(plp, plprate, 
                                           filtTime=beatFiltTime, 
                                           thresh=beatThresh)
    dbPreds = beatTimes(peakIndices, plprate)
    
    if snap:
        PI = np.squeeze(PI)
        PI = np.divide(PI, hr)
        dbPreds = snapToOnsets(dbPreds, PI)
 
    return dbPreds


# another post-processing option (not used)
def cutStartPLP(path, cutTime=1.0, **kwargs):
    preds = getPLP(path, **kwargs)
    preds = np.where(preds < cutTime, 0, preds)
    return preds


# get predicted beat and downbeat times
def beatTracker(inputFile):
    beats = getPLP(inputFile, sr=22050, bands = 10, 
                   bandWeights=[4, 3, 2, 1, 0.5, 0.5, 0.5, 0.5, 2, 4], 
                   tType='both', combWeight=0.5, BPMres=1, 
                   constrain=True, constrainFactor=0.3, salWeight=0.3, 
                   snap=True)
    downbeats = getDBPLP(inputFile, sr=22050, bands = 10, 
                         bandWeights=[4, 3, 2, 1, 0.5, 0.5, 0.5, 0.5, 2, 4], 
                         tType='both', combWeight=1.5, minBPM=15, 
                         maxBPM=80, BPMres=0.5,
                         constrain=True, constrainFactor=0.3, salWeight=1, 
                         ncWeight=1.5, salWeights=[0.4, 1, -0.4], winTime=15.0, 
                         PLPwinTime=12.0, snap=True)   
    return beats, downbeats



    
