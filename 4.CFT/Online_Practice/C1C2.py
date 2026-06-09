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



class OriginalSignal(ContinuousSignal):
    def values(self):
        t = self.t
        return 2*np.sin(14*np.pi*t) - np.sin(2*np.pi*t) * (4*np.sin(2*np.pi*t)*np.sin(14*np.pi*t) - 1)


class DecomposedSignal(ContinuousSignal):
    def values(self):
        t = self.t
        # f(t) = sin(2πt) + sin(10πt) + sin(18πt)
        return np.sin(2*np.pi*t) + np.sin(10*np.pi*t) + np.sin(18*np.pi*t)

t = np.linspace(0, 1, 5000, endpoint=False)

gen = SignalGenerator(t)

original = CompositeSignal(t)
original.add_component(gen.sine(2,7))
original.add_component(gen.sine(-1,1)*((gen.sine(4,1)*gen.sine(1,7))-1))


# -----------------------------
# Time axis (choose 1 second so 1,5,9 Hz fit integer cycles)
# -----------------------------


orig = original
decomp = DecomposedSignal(t)

x_orig = orig.values()
x_decomp = decomp.values()

# -----------------------------
# Plot time-domain signals
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(t, x_orig, label="Original f(t)", linewidth=1.5)
plt.plot(t, x_decomp, "--", label="Sum of components", linewidth=1.5)
plt.xlabel("t (sec)")
plt.ylabel("Amplitude")
plt.title("Original vs Decomposed Reconstruction")
plt.legend()
plt.grid(True)
plt.tight_layout()

# -----------------------------
# Compute CFT magnitude using your CFTAnalyzer (numerical integration)
# -----------------------------
freqs = np.linspace(-20, 20, 2001)  # 0 to 20 Hz (step 0.01 Hz)
analyzer = CFTAnalyzer(orig, t, freqs)
realX, imagX = analyzer.compute_cft()
mag = np.sqrt(realX**2 + imagX**2)

# Find top 3 peak frequencies (simple approach)
# take top 10, then filter unique-ish peaks
top_idx = np.argsort(mag)[-10:][::-1]
peaks = []
for i in top_idx:
    f = freqs[i]
    if all(abs(f - pf) > 0.2 for pf in peaks):  # avoid near-duplicates
        peaks.append(f)
    if len(peaks) == 3:
        break

print("Estimated dominant frequencies (Hz):", peaks)

# Plot magnitude spectrum
plt.figure(figsize=(10, 4))
plt.plot(freqs, mag, linewidth=1.5)
for pf in peaks:
    plt.axvline(pf, linestyle="--", linewidth=1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X(f)| (magnitude)")
plt.title("CFT Magnitude Spectrum (Peaks should be near 1, 5, 9 Hz)")
plt.grid(True)
plt.tight_layout()

plt.show()