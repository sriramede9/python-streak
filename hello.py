# understanding pydantic
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 76, 7])
print(arr)

two_dimensional_arr = np.array([[1, 2], [2, 3]])
print(two_dimensional_arr)
print(two_dimensional_arr.shape)
# this shape would be one column , two rows so 2, 2? means I have two rows and two columns

# slice
print(arr[:2])  # [1,2]
print(arr[::2])  # alternative element [1,3,5,76]
print(arr[5:,])  # skip first 5 elements [6,76,7]

# how to make it n dimensional array
new_arr = np.array([1,2,3,4,5],ndmin=5) # crates a 5D array [[[[[1,2,3,4,5]]]]]
#NumPy won't remove dimensions. It only adds singleton (size 1) dimensions if needed
