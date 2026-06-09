import numpy as np

a=np.array([[1,2,3,4],[5,6,7,8]]);
a=a[a<7]
print(a)

x=np.arange(1,25).reshape(2,12)
print(x)
y=np.hsplit(x,3)
print(y)

Z=np.random.uniform(1,10,(8,8))
print(Z)