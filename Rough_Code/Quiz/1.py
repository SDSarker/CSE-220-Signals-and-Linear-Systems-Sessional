import numpy as np
data = np.array ([3 , 6, 1, 2, 4, 7, 8])
data = data[ data % 2 == 0 ]
data = data / 2
result = data ** 2
print( result .sum())