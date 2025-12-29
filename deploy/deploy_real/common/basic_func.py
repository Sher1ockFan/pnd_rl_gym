import numpy as np
import math


class LowPassFilter:
    """
    Second-order low-pass filter implementation.
    Converted from C++ LowPassFilter class.
    """
    
    def __init__(self, cut_off_freq, damp_ratio, d_time, n_filter):
        """
        Initialize the low-pass filter.
        
        Args:
            cut_off_freq: Cut-off frequency in Hz
            damp_ratio: Damping ratio
            d_time: Time step (dt) in seconds
            n_filter: Number of filters (size of signal vectors)
        """
        self.dt_ = d_time
        self.n_filter_ = n_filter
        
        # Initialize state vectors
        self.sig_in_1_ = np.zeros(n_filter)
        self.sig_in_2_ = np.zeros(n_filter)
        self.sig_out_1_ = np.zeros(n_filter)
        self.sig_out_2_ = np.zeros(n_filter)
        
        # Calculate filter coefficients
        freq_in_rad = 2.0 * math.pi * cut_off_freq
        c = 2.0 / self.dt_
        sqr_c = c * c
        sqr_w = freq_in_rad * freq_in_rad
        
        # Calculate numerator coefficients (b)
        b2_ = sqr_c + 2.0 * damp_ratio * freq_in_rad * c + sqr_w
        b1_ = -2.0 * (sqr_c - sqr_w)
        b0_ = sqr_c - 2.0 * damp_ratio * freq_in_rad * c + sqr_w
        
        # Calculate denominator coefficients (a)
        a2_ = sqr_w
        a1_ = 2.0 * sqr_w
        a0_ = sqr_w
        
        # Normalize coefficients by b2
        a2_ /= b2_
        a1_ /= b2_
        a0_ /= b2_
        
        b1_ /= b2_
        b0_ /= b2_
        b2_ = 1.0
        
        # Store normalized coefficients
        self.a0_ = a0_
        self.a1_ = a1_
        self.a2_ = a2_
        self.b0_ = b0_
        self.b1_ = b1_
        self.b2_ = b2_
    
    def update(self, signal_in):
        """
        Apply the low-pass filter to the input signal.
        
        Args:
            signal_in: Input signal (numpy array of size n_filter)
            
        Returns:
            Filtered output signal (numpy array of size n_filter)
        """
        # Ensure input is numpy array
        if not isinstance(signal_in, np.ndarray):
            signal_in = np.array(signal_in)
        
        # Apply filter: y[n] = a0*x[n] + a1*x[n-1] + a2*x[n-2] 
        #                      - b1*y[n-1] - b2*y[n-2]
        signal_out = (self.a0_ * signal_in + 
                      self.a1_ * self.sig_in_1_ + 
                      self.a2_ * self.sig_in_2_ -
                      self.b1_ * self.sig_out_1_ - 
                      self.b2_ * self.sig_out_2_)
        
        # Update state history
        self.sig_in_2_ = self.sig_in_1_.copy()
        self.sig_in_1_ = signal_in.copy()
        self.sig_out_2_ = self.sig_out_1_.copy()
        self.sig_out_1_ = signal_out.copy()
        
        return signal_out
    
    def reset(self):
        """Reset filter state to zero."""
        self.sig_in_1_ = np.zeros(self.n_filter_)
        self.sig_in_2_ = np.zeros(self.n_filter_)
        self.sig_out_1_ = np.zeros(self.n_filter_)
        self.sig_out_2_ = np.zeros(self.n_filter_)





