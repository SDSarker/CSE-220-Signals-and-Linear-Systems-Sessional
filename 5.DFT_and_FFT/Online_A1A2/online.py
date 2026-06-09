import numpy as np


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


# Example usage
x = 33
y = 444

#converting to digit arrays(discrete signal)
x_digits = [int(digit) for digit in str(x)]

y_digits = [int(digit) for digit in str(y)]

signal_x= DiscreteSignal(np.array(x_digits,dtype=np.complex128))
signal_y= DiscreteSignal(np.array(y_digits,dtype=np.complex128))

N=max(len(signal_x),len(signal_y))
padding= 2*N -1

signal_x=signal_x.pad(padding)
signal_y=signal_y.pad(padding)

analyzer=FastFourierAnalyzer()
X=analyzer.compute_dft(signal_x)
Y=analyzer.compute_dft(signal_y)

Z=X*Y

z= analyzer.compute_idft(Z)

result=np.round(np.real(z)).astype(int)

actual_length= len(x_digits)+len(y_digits) -1
result=result[:actual_length]

sum_res=0
carry=0
digits=[]
for i in result[::-1] :
    sum_res = i+carry
    digits.append(sum_res%10)
    carry=sum_res//10

while(carry>0):
    digits.append(carry%10)
    carry=carry//10

print(''.join(map(str,reversed(digits))))