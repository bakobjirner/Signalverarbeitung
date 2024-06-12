from os import path as osp
import numpy as np
from utils import ideal_bandpass, from_wav, to_wav, plot_freq, plot_am_freq, check_arrays

# Your solution starts here.
def frequency_demodulation(high_freq_signal, rate, carrier_freq, bandwidth, freq_delta):   
    """
    Demodulates a high-frequency signal by extracting the original low-frequency signal using frequency demodulation.

    Args:
        high_freq_signal (ndarray): The high-frequency signal to be demodulated.
        rate (float): The sampling rate of the high-frequency signal.
        carrier_freq (float): The carrier frequency used in modulation.
        bandwidth (float): The maximum frequency allowed in the filtered signals.
        freq_delta (float): The frequency offset or deviation used in modulation.

    Returns:
        ndarray: The demodulated low-frequency signal.

    """
    
    return np.zeros_like(high_freq_signal) # TODO: 4a
    
def frequency_modulation(low_freq_signal, rate, carrier_freq, bandwidth, freq_delta):
    """
    Modulates a low-frequency signal onto a carrier frequency using frequency modulation (FM).

    Args:
        low_freq_signal (ndarray): The input low-frequency signal to be modulated.
        rate (float): The sampling rate of the low-frequency signal.
        carrier_freq (float): The frequency of the carrier signal.
        bandwidth (float): The maximum frequency allowed in the filtered low-frequency signal.
        freq_delta (float): The frequency offset or deviation.

    Returns:
        ndarray: The frequency-modulated signal.

    """
    
    return np.zeros_like(low_freq_signal[:-1]) # TODO: 4b
# Your solution ends here.

def main():
    """You may adapt the center-frequency to listen to all channels."""
    
    # load the antenna signal
    rate, antenna_signal = from_wav('antenna_signal_fm.wav')
    plot_freq(
        antenna_signal,
        rate=rate,
        title='4a) Antenna HF FM Signal',
        block=False,
    )
    
    # demodulate
    received_signal = frequency_demodulation(
        high_freq_signal=antenna_signal,
        rate=rate,
        carrier_freq=40e3, # 4a) Change this value receive other channels.
        bandwidth=5e3,
        freq_delta=5e3,
    )
    plot_freq(
        received_signal,
        rate,
        xlim=10e3,
        title='4a) Demodulated LF Signal',
        block=False,
    )
    
    # write the received signal to the file system
    to_wav('received_signal_fm_4a.wav', rate=rate, x=received_signal)
    
    # load low frequency signal
    rate_bach, low_freq_bach = from_wav(osp.join('audio','bach.wav'))
    
    
    # modulate signal
    carrier_freq = 50e3
    high_freq_bach = frequency_modulation(
        low_freq_signal=low_freq_bach,
        rate=rate_bach,
        carrier_freq=carrier_freq,
        bandwidth=5e3,
        freq_delta=5e3,
    )
    
    # add some white noise
    high_freq_bach += 0.001 * np.random.randn(high_freq_bach.shape[0])
    plot_freq(
        high_freq_bach,
        rate=rate_bach,
        title='4b) Modulated HF Signal',
        block=False,
    )
    
    # demodulate
    received_signal = frequency_demodulation(
        high_freq_signal=high_freq_bach,
        rate=rate_bach,
        carrier_freq=carrier_freq,
        bandwidth=5e3,
        freq_delta=5e3,
    )
    
    # Compare original LF signal to demodulated version.
    plot_freq(
        low_freq_bach,
        rate_bach,
        xlim=10e3,
        title='4b) Original LF Signal',
        block=False,
    )
    plot_freq(
        received_signal,
        rate_bach,
        xlim=10e3,
        title='4b) Demodulated LF Signal',
        block=False,
    )
    
    # write the received signal to the file system
    to_wav('bach_demodulated_fm_4b.wav', rate=rate, x=received_signal)
    
    input('Press ENTER to close all plots and quit.')
    
if __name__ == '__main__':
    main()
