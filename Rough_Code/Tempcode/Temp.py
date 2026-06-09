import numpy as np
import matplotlib.pyplot as plt


def time_shift_signal(x : np.ndarray, k : int) -> np.ndarray:  #x-k
    temp=np.zeros_like(x)
    if(k>0):
        temp[k:]=x[:-k]
    elif(k<0):
        temp[:k]=x[-k:]
    else:
        temp[:]=x[:]
    
    return temp

def time_scale_signal(x : np.ndarray, k : int) -> np.ndarray:   #xk
    temp=np.zeros_like(x)

    second_half=x[INF::k]
    # 1. 1. 0. 0. 0.
    x=np.flip(x)
    first_half=x[INF+k::k]
    first_half=np.flip(first_half)
    # 0.  0.  0.  0.5

    temp[INF:INF+len(second_half)]=second_half[:]
    
    temp[INF-len(first_half):INF]=first_half[:]
    
    return temp

def time_scale_signal2(x : np.ndarray, k : int) -> np.ndarray:   #x/k
    if k==1 :
        return x
    temp=np.zeros_like(x)
    
    second_half=x[INF:]
    x=np.flip(x)
    first_half=x[INF+1:]
    first_half=np.flip(first_half)
    

    right_slot=temp[INF::k]
    temp[INF::k]=second_half[:len(right_slot):]
    
    left_slot=temp[:INF:k]
    temp[INF-k::-k]=first_half[INF-1:INF-len(left_slot):-1]
    return temp

def time_scale_signal_interpolate(x: np.ndarray, k: int) -> np.ndarray:

    temp = np.zeros_like(x, dtype=float)
    center = 8

    for i in range(len(x)):
        # New position relative to center
        new_idx = center + (i - center) * k
        if 0 <= new_idx < len(x):
            temp[new_idx] = x[i]

    for i in range(len(x)):
        idx_a = center + (i - center) * k
        idx_b = center + (i + 1 - center) * k
        
        
        if 0 <= idx_a < len(x) and 0 <= idx_b < len(x):
            avg_value = (temp[idx_a] + temp[idx_b]) / 2
            
            temp[idx_a + 1 : idx_b] = avg_value
            
    return temp


def time_reverse_signal(x : np.ndarray) -> np.ndarray:
    return np.flip(x)


def odd_even_decomposition(x : np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    y=np.flip(x)
    odd=(x-y)/2
    even=(x+y)/2
    return odd, even
