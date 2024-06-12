from os import path as osp
import numpy as np
from utils import ideal_bandpass, from_wav, to_wav, plot_freq, plot_am_freq, check_arrays

# Your solution starts here.
def hilbert_trafo(x_time):
    """
    Applies the Hilbert transform to the input signal to obtain its analytic representation.

    Args:
        x_time (ndarray): The input signal in the time domain.

    Returns:
        ndarray: The transformed input signal.

    """
    
    return np.zeros_like(x_time) # TODO: 3a

def ssb_demodulation(high_freq_signal, rate, carrier_freq, bandwidth, upper_side_band=True):
    """
    Demodulates a Single Sideband (SSB) modulated signal to recover the original low-frequency signal.

    Args:
        high_freq_signal (ndarray): The SSB-modulated high-frequency signal to be demodulated.
        rate (float): The sampling rate of the high-frequency signal.
        carrier_freq (float): The carrier frequency used in modulation.
        bandwidth (float): The maximum frequency allowed in the filtered signals.
        upper_side_band (bool, optional): Determines whether the upper sideband was used during modulation.
                                         Defaults to True.

    Returns:
        ndarray: The demodulated low-frequency signal.

    """
    
    return np.zeros_like(high_freq_signal) # TODO: 3b
    
def ssb_modulation(low_freq_signal, rate, carrier_freq, bandwidth, upper_side_band=True):
    """
    Performs Single Sideband (SSB) modulation on a low-frequency signal using Suppressed Carrier SSB modulation.

    Args:
        low_freq_signal (ndarray): The input low-frequency signal to be modulated.
        rate (float): The sampling rate of the low-frequency signal.
        carrier_freq (float): The frequency of the carrier signal.
        bandwidth (float): The maximum frequency allowed in the filtered low-frequency signal.
        upper_side_band (bool, optional): Determines whether the upper sideband is used. Defaults to True.

    Returns:
        ndarray: The SSB-modulated signal.

    """
    
    return np.zeros_like(low_freq_signal) # TODO: 3c
# Your solution ends here.

def main():
    """You may adapt the center-frequency to listen to all channels."""
    
    rate = 192e3
    x = ideal_bandpass(
        np.random.randn(int(rate)),
        rate,
        min_freq=0.1e3,
        max_freq=1e3
    )
    check_arrays('Exercise 3', ['a) Hilbert Transformation'], [hilbert_trafo(hilbert_trafo(x))], [-x])
    
    # load the antenna signal
    rate, antenna_signal = from_wav('antenna_signal_ssb.wav')
    plot_am_freq(
        antenna_signal,
        rate=rate,
        title='3b) Antenna HF FM Signal',
        block=False,
    )
    
    # demodulate
    received_signal = ssb_demodulation(
        high_freq_signal=antenna_signal,
        rate=rate,
        carrier_freq=48e3, # 3b) Change this value receive other channels.
        bandwidth=9e3,
        upper_side_band=False, # 3b) Change this value receive other channels.
    )
    plot_am_freq(
        received_signal,
        rate,
        xlim=10e3,
        title='3b) Demodulated LF Signal',
        block=False,
    )
    
    # write the received signal to the file system
    to_wav('received_signal_ssb_3b.wav', rate=rate, x=received_signal)
    
    # load low frequency signal
    rate_bach, low_freq_bach = from_wav(osp.join('audio','bach.wav'))
    
    
    # modulate signal
    carrier_freq = 50e3
    high_freq_bach = ssb_modulation(
        low_freq_signal=low_freq_bach,
        rate=rate_bach,
        carrier_freq=carrier_freq,
        bandwidth=9e3,
        upper_side_band=True,
    )
    
    # add some white noise
    high_freq_bach += 0.001 * np.random.randn(high_freq_bach.shape[0])
    plot_freq(
        high_freq_bach,
        rate=rate_bach,
        title='3c) Modulated HF Signal',
        block=False,
    )
    
    # demodulate
    received_signal = ssb_demodulation(
        high_freq_signal=high_freq_bach,
        rate=rate_bach,
        carrier_freq=carrier_freq,
        bandwidth=9e3,
        upper_side_band=True,
    )
    
    # Compare original LF signal to demodulated version.
    plot_freq(
        low_freq_bach,
        rate_bach,
        xlim=10e3,
        title='3c) Original LF Signal',
        block=False,
    )
    plot_freq(
        received_signal,
        rate_bach,
        xlim=10e3,
        title='3c) Demodulated LF Signal',
        block=False,
    )
    
    # Write the received signal to the file system.
    to_wav('bach_demodulated_ssb_3c.wav', rate=rate, x=received_signal)
    
    input('Press ENTER to close all plots and quit.')
    
if __name__ == '__main__':
    main()
