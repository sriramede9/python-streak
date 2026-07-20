import os
import numpy as np

api_key = os.environ.get('API_KEY')

print(api_key)

arr = np.array([[1,2,3,"str",4],[5,6,7,"str2",9]])

print(arr.dtype)
print(arr[1,2])