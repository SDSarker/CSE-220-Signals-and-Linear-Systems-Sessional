import cv2
import numpy as np
import math


class DiscreteSignal:
    """
    Represents a discrete-time signal.
    """
    def __init__(self, data):
        # Ensure data is a numpy array, potentially complex
        self.data = np.array(data, dtype=np.complex128)

    def __len__(self):
        return len(self.data)
        
    def pad(self, new_length):
        """
        Zero-pad or truncate signal to new_length.
        Returns a new DiscreteSignal object.
        """
        n=len(self.data)

        if new_length>n :
            temp=np.zeros(new_length)
            temp[:n]=self.data[:]
        else:
            temp=self.data[:new_length]

        return DiscreteSignal(temp)

    def interpolate(self, new_length):
        """
        Resample signal to new_length using linear interpolation.
        Required for Task 4 (Drawing App).
        """
        old_len=len(self.data)
        
        if old_len ==0:
            return DiscreteSignal(np.zeros(new_length, dtype=np.complex128))
        elif old_len== 1:
            return DiscreteSignal(np.ones(new_length,dtype=np.complex128) * self.data[0])
        
        old_idx=np.linspace(0,1,old_len)
        new_idx=np.linspace(0,1,new_length)

        r=np.interp(new_idx,old_idx,np.real(self.data))
        i=np.interp(new_idx,old_idx,np.imag(self.data))


        return DiscreteSignal(r+1j*i)


class DFTAnalyzer:
    """
    Performs Discrete Fourier Transform using O(N^2) method.
    """
    def compute_dft(self, signal: DiscreteSignal):
        """
        Compute DFT using naive summation.
        Returns: numpy array of complex frequency coefficients.
        """

        N = len(signal)
        n=np.arange(N)   #row
        k=np.arange(N).reshape((N,1))        #column

        e=np.exp(-2j*np.pi*k*n/N)

        return np.dot(e,signal.data)


    def compute_idft(self, spectrum):
        """
        Compute Inverse DFT using naive summation.
        Returns: numpy array (time-domain samples).
        """
        N = len(spectrum)
        k=np.arange(N)
        n=np.arange(N).reshape((N,1))

        e=np.exp(2j*np.pi*k*n/N)

        return np.dot(e,spectrum)/N
    
class FastFourierAnalyzer(DFTAnalyzer):
    
    def compute_dft(self, signal: DiscreteSignal):
        x= signal.data
        N=len(x)

        if (N & (N-1)) !=0:
            next_power_of_2=1<<(N-1).bit_length()
            x=signal.pad(next_power_of_2).data

        return self.fft(x)
    
    def fft(self, x):
        N=len(x)

        if N<=1:
            return x

        even=self.fft(x[::2])
        odd=self.fft(x[1::2])

        T = np.exp(-2j *np.pi * np.arange(N // 2) / N)*odd

        return np.concatenate([even + T, even - T])

    def compute_idft(self, spectrum):
        
        X=np.asarray(spectrum, dtype=np.complex128)
        N=len(X)

        X_conj=np.conjugate(X)

        x_fft=self.fft(X_conj)

        return np.conjugate(x_fft)/N


def fft(x):
    """
    Compute 1D FFT
    """
    pass

def ifft(X):
    """
    Compute 1D inverse FFT using the FFT function
    """
    pass

def reconstruct_image_using_fft(original_path, shifted_path, output_path):
    
    original_img = cv2.imread(original_path)
    shifted_img = cv2.imread(shifted_path)

    if original_img is None or shifted_img is None:
        print("Error: Could not load images.")
        return

    if original_img.shape != shifted_img.shape:
        print("Error: Image dimensions do not match.")
        return
    
    # Convert the original and shifted color images to grayscale.
    orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    shift_gray = cv2.cvtColor(shifted_img, cv2.COLOR_BGR2GRAY)
    
    reconstructed_img = np.zeros_like(shift_gray)
    
    i=0
    for row in orig_gray:
        x=DiscreteSignal(orig_gray[i])
        y=DiscreteSignal(shift_gray[i])
        
        
        analyzer=FastFourierAnalyzer()
        X=analyzer.compute_dft(x)
        Y=analyzer.compute_dft(y)
        
        Z=X*np.conjugate(Y)
        
        z=analyzer.compute_idft(Z)
        z_real=np.real(z)
        
        shift=np.argmax(z_real)
        
        shifted_row=np.roll(shift_gray[i],shift)
        reconstructed_img[i]=shifted_row
        
        
        i+=1
        
        
    
    

    print("Reconstructing image using manual FFT...")
    
    i=0
    for row in reconstructed_img:
       j=0
       for x in row:
           if reconstructed_img[i][j]!=orig_gray[i][j]:
               print("Mismatched")
               return

    print("Reconstructed Successfully")
    #implement the rest

    

    cv2.imwrite(output_path, reconstructed_img)
    


if __name__ == "__main__":
    reconstruct_image_using_fft("original_image.png", "shifted_image.jpg", "new.jpg")