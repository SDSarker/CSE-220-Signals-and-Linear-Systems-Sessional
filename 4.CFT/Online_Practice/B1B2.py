import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Abstract Base Class for Continuous-Time Signals
# =====================================================
class ContinuousSignal:
    """
    Abstract base class for all continuous-time signals.
    Every signal must be defined over a time axis t.
    """

    def __init__(self, t):
        self.t = t

    def values(self):
        """
        Returns the signal values evaluated over time axis t.
        Must be implemented by subclasses.
        """

        return np.zeros_like(self.t)
        

    def plot(self, title="Signal"):
        """
        Plot the signal in the time domain.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(self.t, self.values())
        plt.xlabel("Time (t)")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.grid(True)
        plt.show()


# =====================================================
# Signal Generator Class
# =====================================================
class SignalGenerator(ContinuousSignal):
    """
    Generates various continuous-time signals.
    Each method returns a numpy array of signal samples.
    """

    def sine(self, amplitude, frequency):
        """Generate a sine wave."""
        return amplitude*np.sin(2*np.pi*frequency*self.t)

    def cosine(self, amplitude, frequency):
        """Generate a cosine wave."""
        return amplitude*np.cos(2*np.pi*frequency*self.t)

    def square(self, amplitude, frequency):
        """Generate a square wave using sign of sine."""
        return amplitude* np.sign(np.sin(2*np.pi*frequency*self.t))

    def sawtooth(self, amplitude, frequency):
        """Generate a sawtooth wave."""
        return amplitude*2*((frequency*self.t)-np.floor(0.5+frequency*self.t))

    def triangle(self, amplitude, frequency):
        """Generate a triangle wave."""
        return (2*amplitude/np.pi)*np.arcsin(np.sin(2*np.pi*frequency*self.t))

    def cubic(self, coefficient):
        """Generate a cubic polynomial signal."""
        return coefficient*self.t**3

    def parabolic(self, coefficient):
        """Generate a parabolic signal."""
        return coefficient*self.t**2

    def rectangular(self, width):
        """Generate a rectangular window centered at t=0."""
        return np.where(np.abs(self.t)<=(width/2),1,0)

    def pulse(self, start, end):
        """Generate a finite pulse active between start and end."""
        return np.where((self.t>=start) & (self.t<=end),1,0)


# =====================================================
# Composite Signal Class
# =====================================================
class CompositeSignal(ContinuousSignal):
    """
    Combines multiple signals into a single composite signal.
    """

    def __init__(self, t):
        super().__init__(t)
        self.components = []

    def add_component(self, signal):
        """
        Add a signal component to the composite signal.
        """
        self.components.append(signal)

    def values(self):
        """
        Sum all signal components.
        """
        x=np.zeros_like(self.t)
        for s in self.components:
            x+=s

        return x            


# =====================================================
# Continuous Fourier Transform Analyzer
# =====================================================
class CFTAnalyzer:
    """
    Computes the Continuous Fourier Transform (CFT)
    using numerical integration (np.trapz).
    """

    def __init__(self, signal, t, frequencies):
        self.signal = signal
        self.t = t
        self.frequencies = frequencies

    def compute_cft(self):
        """
        Compute real and imaginary parts of the CFT.
        """

        y=self.signal.values() #x(t)
        real_parts=[]
        im_parts=[]

        for f in self.frequencies:
            r= np.trapezoid(y*np.cos(2*np.pi*f*self.t),self.t)
            i= -np.trapezoid(y*np.sin(2*np.pi*f*self.t),self.t)
            real_parts.append(r)
            im_parts.append(i)

        return np.array(real_parts),np.array(im_parts)

    def plot_spectrum(self):
        """
        Plot magnitude spectrum of the signal.
        """
        real, img=self.compute_cft()
        magnitude=np.sqrt(real**2 + img**2)
        plt.figure(figsize=(10,4))
        plt.plot(self.frequencies, magnitude)
        plt.title("Magnitude Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        plt.show()
        return magnitude


# =====================================================
# Inverse Continuous Fourier Transform
# =====================================================
class InverseCFT:
    """
    Reconstructs time-domain signal using ICFT.
    """

    def __init__(self, spectrum, frequencies, t):
        self.spectrum = spectrum
        self.frequencies = frequencies
        self.t = t

    def reconstruct(self):
        """
        Perform inverse CFT using numerical integration.
        """
        real=self.spectrum[0]
        img=self.spectrum[1]

        reconstruct=[]
        for ti in self.t:
            x=np.trapezoid(real*np.cos(2*np.pi*self.frequencies*ti),self.frequencies) - np.trapezoid(img*np.sin(2*np.pi*self.frequencies*ti),self.frequencies)

            reconstruct.append(x)


        return np.array(reconstruct)


# Piecewise Signal (Given in the figure)
# =====================================================

class PiecewiseSignal(ContinuousSignal):
    """
    f(t) defined over [-3, 3] as:
      - For -3 <= t <= -1 : (t + 3)^2   (parabola)
      - For -1 <  t <  1 : 5 - |t|      (line segments)
      - For  1 <= t <=  3 : (t - 3)^2   (parabola)
      - Otherwise : 0
    """

    def values(self):
        t = self.t
        x = np.zeros_like(t, dtype=float)

        # Left parabola: [-3, -1]
        mask1 = (t >= -3) & (t <= -1)
        x[mask1] = (t[mask1] + 3) ** 2

        # Middle: (-1, 1)
        mask2 = (t > -1) & (t < 1)
        x[mask2] = 5 - np.abs(t[mask2])

        # Right parabola: [1, 3]
        mask3 = (t >= 1) & (t <= 3)
        x[mask3] = (t[mask3] - 3) ** 2

        return x
    

# =====================================================
# Parseval Verification
# =====================================================

class ParsevalVerifier:
    """
    Verifies Parseval's theorem numerically:
      ∫ |x(t)|^2 dt  ≈  ∫ |X(f)|^2 df
    """

    def __init__(self, t, x_t, f, X_f):
        self.t = t
        self.x_t = x_t
        self.f = f
        self.X_f = X_f  # (realX, imagX)

    def time_domain_energy(self):
        return np.trapezoid(np.abs(self.x_t) ** 2, self.t)

    def freq_domain_energy(self):
        realX, imagX = self.X_f
        mag2 = realX**2 + imagX**2
        return np.trapezoid(mag2, self.f)

    def report(self):
        Et = self.time_domain_energy()
        Ef = self.freq_domain_energy()
        abs_err = np.abs(Et - Ef)
        rel_err = abs_err / (np.abs(Et) + 1e-12)

        print("=====================================================")
        print("Parseval Verification")
        print("=====================================================")
        print(f"Time-domain energy  Et = ∫|x(t)|^2 dt  = {Et:.6f}")
        print(f"Freq-domain energy  Ef = ∫|X(f)|^2 df  = {Ef:.6f}")
        print(f"Absolute error      |Et - Ef|          = {abs_err:.6e}")
        print(f"Relative error      |Et - Ef|/|Et|     = {rel_err:.6e}")
        print("=====================================================")

        return Et, Ef, abs_err, rel_err


# =====================================================
# Main Execution
# =====================================================

if __name__ == "__main__":
    # -----------------------------
    # Time axis (function is nonzero only on [-3, 3])
    # -----------------------------
    t = np.linspace(-3, 3, 6001)
    x_signal = PiecewiseSignal(t)
    x_t = x_signal.values()

    # Plot time-domain function
    x_signal.plot(title="Given Piecewise Function f(t)")

    # -----------------------------
    # Frequency axis (choose wide enough for good approximation)
    # -----------------------------
    f = np.linspace(-30, 30, 6001)

    # Compute CFT
    analyzer = CFTAnalyzer(x_signal, t, f)
    realX, imagX = analyzer.compute_cft()

    # Plot magnitude spectrum
    analyzer.plot_spectrum()

    # -----------------------------
    # Verify Parseval's theorem
    # -----------------------------
    verifier = ParsevalVerifier(t, x_t, f, (realX, imagX))
    verifier.report()

    # -----------------------------
    # Optional: Reconstruct using ICFT and compare
    # -----------------------------
    icft = InverseCFT((realX, imagX), f, t)
    x_rec = icft.reconstruct()

    plt.figure(figsize=(10, 4))
    plt.plot(t, x_t, label="Original x(t)")
    plt.plot(t, x_rec, "--", label="Reconstructed x(t)")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.title("Original vs Reconstructed (ICFT)")
    plt.grid(True)
    plt.legend()
    plt.show()