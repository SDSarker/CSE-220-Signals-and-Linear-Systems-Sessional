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
        

class SuperSignal:
    def __init__(self):
        self.components = []

    def add(self, signal: Signal, coefficient=1.0):
        self.components.append((coefficient, signal))

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
    
    def output_super(self, ss:SuperSignal):
        res = Signal(10)
        for i in ss.components:
            coeff= i[0]
            sig=i[1]
            x=sig.multiply(coeff)
            res=res.add(x)

        res.plot("Res")

        return self.output(res)





# Todo: Define Signal class
        

        
# Todo: Define LTI class

if __name__ == "__main__":
    INF = 10

    # Component signals
    x1 = Signal(INF)
    x1.set_value_at_time(0, 1)

    x2 = Signal(INF)
    x2.set_value_at_time(2, 1)

    # Todo: Create SuperSignal: x(n) = 2*x1(n) - x2(n)

    ss=SuperSignal()
    ss.add(x1,2)
    ss.add(x2,-1)

    # Impulse response
    h = Signal(INF)
    h.set_value_at_time(0, 1)
    h.set_value_at_time(1, 0.5)

    h.plot()

    system = LTI_System(h)

    # Todo: Output using superposition

    y=system.output_super(ss)

    y.plot()
    plt.show()