import numpy as np

x = np.array([ 55.5806, 53.3801,60.7708])
print('{:.4f}±{:.3f}'.format(np.mean(x),np.std(x)))
