import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt

def rectifier(x):
    """Applies rectification to a given input signal.
    
    Maps negative values to zero. Positive values are returned unchanged.

    Args:
        x (array-like): The input signal.

    Returns:
        array-like: The rectified signal.

    Example:
        >>> x = np.array([-1, 2, -3, 4, -5])
        >>> rectifier(x)
        array([0, 2, 0, 4, 0])

    """
    return x * (x > 0)
    
def ideal_bandpass(x_time, rate, min_freq=None, max_freq=None):
    """
    Applies ideal bandpass filtering to a time-domain signal.

    Args:
        x_time (ndarray): Time-domain signal.
        rate (int): Sampling frequency in Hz.
        min_freq (float, optional): Minimum frequency of the passband in Hz. Defaults to -inf.
        max_freq (float, optional): Maximum frequency of the passband in Hz. Defaults to inf.

    Returns:
        ndarray: Filtered time-domain signal.

    Examples:
        >>> # Applies a bandpass filter:
        >>> x_filtered = ideal_bandpass(x, 192e3, min_freq=1e3, max_freq=5e3)
        
        >>> # Can also be used as a lowpass:
        >>> x_filtered = ideal_bandpass(x, 192e3, max_freq=5e3)     
    """
    length = x_time.shape[0]
    x_freq = np.fft.fftshift(np.fft.fft(x_time))
    freqs = np.abs(np.arange(length) - length//2)
    
    a = x_time.shape[0] / rate
    
    if min_freq is not None:
        x_freq[freqs < a * min_freq] = 0
    if max_freq is not None:
        x_freq[freqs > a * max_freq] = 0
    
    x_time_filtered = np.fft.ifft(np.fft.ifftshift(x_freq))
    return x_time_filtered

def from_wav(filename):
    """Read a wave file.

    A wrapper function for scipy.io.wavfile.read
    that also includes int16 to float [-1,1] scaling.

    Parameters:
        filename (str): File name string.

    Returns:
        fs (int): Sampling frequency in Hz.
        x (ndarray): Array of signal samples normalized to 1.

    Examples:
        >>> fs, x = from_wav('test_file.wav')
    """
    fs, x = wavfile.read(filename)
    return fs, x/0x7fff

def to_wav(filename,rate,x):
    """Write a wave file.

    A wrapper function for `scipy.io.wavfile.write` that also includes int16 scaling and conversion.
    Assumes input `x` has values in the range [-1, 1].

    Parameters:
        filename (str): File name string.
        rate (int): Sampling frequency in Hz.
        x (ndarray): Array of signal samples.

    Examples:
        >>> to_wav('test_file.wav', 8000, x)
    """
    x16 = np.int16(x*0x7fff)
    wavfile.write(filename, rate, x16)
    
def load_signals(paths, n=None):
    """
    Load audio signals from a list of paths and return them as a list.

    Args:
        paths (list of str): List of paths to audio files.
        n (int, optional): Length of the resulting signals. If None, the shortest
            signal will be used as reference. If the desired length is longer than
            the shortest signal, the missing samples will be padded with ending zeros.
            Defaults to None.

    Returns:
        tuple: Tuple containing a list of the sampling rates and a list of the
            audio signals.

    Examples:
        >>> rates, signals = load_signals(["file1.wav", "file2.wav"], 192e3)
    """
    rates, signals = zip(*[from_wav(path) for path in paths])
    m = min([x.shape[0] for x in signals])

    if n is None:
        n = m
    elif n > m:
        signals = [np.pad(x, (0, rectifier(n-x.shape[0]))) for x in signals]
    
    return rates, [x[:n] for x in signals]
    
def plot_freq(signal_time, rate, xlim=None, ylim=None, title=None, block=None):
    """
    Plots the frequency spectrum of a time-domain signal.

    Args:
        signal_time: A 1-dimensional numpy array representing the time-domain signal.
        rate: The sampling frequency of the signal in Hz.
        xlim: Limits of the x-axis (frequency). If None, the default value is set to
            half of the sampling rate (rate//2).
        ylim: Limits of the y-axis. If None, the default value is automatically
            determined based on the maximum absolute values of the real and imaginary
            components of the frequency spectrum.
        title: The figure's title.
        block: Whether to block the program execution until the plot window is closed.
            If not specified, the default behavior is determined by the underlying
            plotting library.

    Examples:
        >>> x = np.sin(2 * np.pi * 10 * np.arange(192e3) / 192e3)
        >>> plot_freq(x, 192e3, xlim=500)
    """
    n = signal_time.shape[0]
    freqs = ((np.arange(n) - n//2) * rate/n)
    signal_freq = np.fft.fftshift(np.fft.fft(signal_time))
    fig, (ax_freq_real, ax_freq_imag) = plt.subplots(2, 1, figsize=(12, 5))
    
    if title is not None:
        plt.suptitle(title)
    
    if xlim is None:
        xlim = rate//2
        
    if ylim is None:
        ylim = 1.25 * max(
            np.abs(signal_freq.real.max()),
            np.abs(signal_freq.imag.max()),
        )
    
    ax_freq_real.plot(freqs, signal_freq.real)
    ax_freq_real.set_ylabel('Real')
    ax_freq_real.set_xlim(-xlim, xlim)
    ax_freq_real.set_ylim(-ylim, ylim)
    
    ax_freq_imag.plot(freqs, signal_freq.imag)
    ax_freq_imag.set_xlabel('Frequency (Hz)')
    ax_freq_imag.set_ylabel('Imaginary')
    ax_freq_imag.set_xlim(-xlim, xlim)
    ax_freq_imag.set_ylim(-ylim, ylim)
    
    plt.show(block=block)
    
def detect_channels_am(x, rate):
    """
    Detects the frequency channels of an amplitude-modulated (AM) signal.

    Args:
        x: A 1-dimensional numpy array representing the AM signal in the time domain.
        rate: The sampling frequency of the signal in Hz.

    Returns:
        An array of frequencies corresponding to the detected channels.

    Examples:
        >>> x = np.sin(2 * np.pi * 10 * np.arange(1000) / 192e3)
        >>> channels = detect_channels_am(x, 192e3)
    """
    n = x.shape[0]
    freqs = ((np.arange(n) - n//2) * rate/n)
    
    x_freq_abs = np.abs(np.fft.fftshift(np.fft.fft(x))[n//2:])
    alpha = 100 * x_freq_abs.std()
    args = np.arange(n//2)[x_freq_abs>alpha]
    return args * (rate/n)
    
def plot_am_freq(am_signal_time, rate, xlim=None, title=None, block=None):
    """
    Plots the frequency spectrum of an amplitude-modulated (AM) signal.
    
    Note that carrier frequencies are ignored when scaling the y-axis.

    Args:
        am_signal_time: A 1-dimensional numpy array representing the AM signal in the time domain.
        rate: The sampling frequency of the signal in Hz.
        xlim: The limits of the x-axis (frequency). If not specified, the default value
            is used based on the input signal.
        title: The figure's title.
        block: Whether to block the program execution until the plot window is closed.
            If not specified, the default behavior is determined by the underlying
            plotting library.

    Returns:
        None

    Examples:
        >>> signal = np.sin(2 * np.pi * 10 * np.arange(1000) / 44100)
        >>> plot_am_freq(signal, 44100, xlim=500)
    """
    channels = detect_channels_am(am_signal_time, rate)
    
    n = am_signal_time.shape[0]
    args = (channels * n/rate).astype(np.int64)
    am_signal_freq = np.fft.fftshift(np.fft.fft(am_signal_time))[n//2:]
    am_signal_freq[args] = 0.
    ylim = 1.25 * max(
        np.abs(am_signal_freq.real.max()),
        np.abs(am_signal_freq.imag.max()),
    )
    
    plot_freq(am_signal_time, rate, xlim=None, ylim=ylim, title=title, block=block)
    
def check_arrays(title, keys, arrays, arrays_true):
    print(f'Checking {title}:')
    for key, array, array_true in zip(keys, arrays, arrays_true):
        passed = array.shape == array_true.shape and np.allclose(array, array_true)
        result = 'passed' if passed  else 'failed'
        print(f'{key}: {result}')
    print()

