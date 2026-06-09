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
        #will trim the 0 from both sides
        # non_zero_indices = np.nonzero(y.signal)[0]
        # min_idx = non_zero_indices[0]
        # max_idx = non_zero_indices[-1]

        # min_t = min_idx - y.INF
        # max_t = max_idx - y.INF

        # new_INF = max(abs(min_t), abs(max_t))

        # trimmed_y = Signal(new_INF)
        # start = min_t + new_INF
        # end = max_t + new_INF + 1

        # trimmed_y.signal[start:end] = y.signal[min_idx : max_idx + 1]

        # return trimmed_y




if __name__ == "__main__":
    with open('input_signal.txt', 'r') as file:
        
        line1 = file.readline().split()
        n_start, n_end = int(line1[0]), int(line1[1])
            
        
        line2 = file.readline().split()
        values = [float(val) for val in line2]

    S=Signal(n_end)
    for i,val in enumerate(values):
        time= i-n_end
        S.set_value_at_time(time,val)

    S.plot("Input Signal x(n)")

    h = Signal(n_end)
    h.set_value_at_time(-2, 0.2)
    h.set_value_at_time(-1, 0.2)
    h.set_value_at_time(-0, 0.2)
    h.set_value_at_time(1, 0.2)
    h.set_value_at_time(2, 0.2)

    h.plot("Impulse Response h(n)")

    # LTI System
    system = LTI_System(h)

    # Output
    y = system.output(S)
    y.plot("Output Signal y(n)")

    plt.show()