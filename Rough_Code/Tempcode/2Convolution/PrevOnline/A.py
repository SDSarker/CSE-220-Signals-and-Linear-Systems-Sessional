import numpy as np
import matplotlib.pyplot as plt

class Signal:
    def __init__(self, INF):
        self.INF=INF
        self.n=2*INF +1
        self.signal=np.zeros(self.n)

    def set_value_at_time(self, t, value):
        if(t<-self.INF or t>self.INF):
            print(f'Error: Time {t} out of bound')
        else:
            self.signal[t+self.INF]=value

    def shift(self, k):
        temp=self.signal
        newINF=self.INF+abs(k)
        newSignal=Signal(newINF)

        #oldSignal er -INF => newSignal er startIdx
        startIdx=newINF-self.INF + k
        #oldSignal er +INF => newSignal er endIdx
        endIdx=startIdx+self.n
        newSignal.signal[startIdx:endIdx]=self.signal[:]
        return newSignal
            

    def add(self, other):
        inf=max(self.INF,other.INF)
        temp=Signal(inf)
        
        #adding self
        start=inf-self.INF
        end=start+self.n
        temp.signal[start:end]=self.signal[:]

        #adding other
        start=inf-other.INF
        end=start+other.n
        temp.signal[start:end]+=other.signal[:]
        
        return temp


    def multiply(self, scalar):
        temp=Signal(self.INF)
        temp.signal=self.signal*scalar
        return temp

    def plot(self, title="Discrete Signal"):
        t=np.arange(-self.INF,self.INF+1)
        
        plt.figure(figsize=(10,4))
        plt.stem(t,self.signal,basefmt=" ", linefmt='b-',markerfmt='bo')
        plt.title(title)
        plt.xlabel("n (time)")
        plt.ylabel("Amplitude")

        plt.grid(True,which="both")
        

class LTI_System:
    def __init__(self, impulse_response: Signal):
        self.h=impulse_response

    def linear_combination_of_impulses(self, input_signal: Signal):
        # Decompose the signal into impulses and corresponding coefficients
        impulses=[]
        coeff=[]

        for i,val in enumerate(input_signal.signal):
            if(val!=0):
                time=i-input_signal.INF

                pulse=Signal(0)
                pulse.set_value_at_time(0,1)
                pulse=pulse.shift(time)

                impulses.append(pulse)
                coeff.append(val)
        return impulses,coeff

    def output(self, input_signal: Signal):
        y=Signal(0)

        impulses,coeff=self.linear_combination_of_impulses(input_signal)
        
        for delta, coeff in zip(impulses,coeff):
            i=np.argmax(delta.signal)
            time=i-delta.INF

            h_shifted=self.h.shift(time)
            h_scaled=h_shifted.multiply(coeff)

            y=y.add(h_scaled)

        return y
        




# Stock Market Prices as a Python List
# price_list = list(map(int, input("Stock Prices: ").split()))
# n = int(input("Window size: "))
# alpha = float(input("Alpha: "))

# You may use the following input for testing purpose
price_list = [10,11,12,9,10,13,15,16,17,18]
n = 4
alpha = 0.8

h=Signal(3)
for i in range(0,n):
    coef=alpha* pow((1-alpha),i)
    h.set_value_at_time(i,coef)

print(h.signal)


inf=len(price_list)

x=Signal(inf)
for i in range(0,inf):
    x.set_value_at_time(i,price_list[i])

system=LTI_System(h)
y=system.output(x)

exsm = []
for i in range(n-1,len(price_list)):
    exsm.append(y.signal[y.INF+i])


# Determine the values after performing Exponential Smoothing
# The length of exsm should be = len(price_list) - n + 1
# exsm = []

print("Exponential Smoothing: " + ", ".join(f"{num:.2f}" for num in exsm))
# Output should be: 11.68, 9.47, 9.82, 12.29, 14.40, 15.62, 16.64, 17.63

