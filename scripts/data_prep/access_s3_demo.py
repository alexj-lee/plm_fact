from dask import array as dda

array = dda.from_zarr('s3://brainform-data/rosetta')
print(array.shape) # (129973008, 1280)
some_slice = array[10:20].compute() # .compute necessary to actually execute lazy op
print(some_slice.shape, some_slice.mean()) # (10, 1280), -0.012938829590406833

import numpy as np
from s3fs.core import S3FileSystem
s3 = S3FileSystem()
x = np.load(s3.open('brainform-data/rosetta_offsets.npy'))

print(x)
# array([212, 294, 119, ..., 127, 228, 295])

print(x.shape)
# (15051,)

assert(np.cumsum(x)[-1]*33 == 129973008) 
# True