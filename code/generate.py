import numpy.random as rn
import cPickle as pickle
import numpy as np
import sktensor as skt

data = rn.poisson(0.2, size=(25, 25, 100))  # 3-mode SPARSE count tensor of size 10 x 8 x 3

subs = data.nonzero()                    # subscripts where the ndarray has non-zero entries   
vals = data[data.nonzero()]              # corresponding values of non-zero entries
sp_data = skt.sptensor(subs,             # create an sktensor.sptensor 
                       vals,
                       shape=data.shape,
                       dtype=data.dtype)

with open('data.dat', 'w+') as f:            # can be stored as a .dat using pickle
    pickle.dump(sp_data, f)

with open('data.dat', 'r') as f:             # can be loaded back in using pickle.load
    tmp = pickle.load(f)
    assert np.allclose(tmp.vals, sp_data.vals)
