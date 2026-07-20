import os
import numpy as np

api_key = os.environ.get("API_KEY")

print(api_key)

arr = np.array([[1, 2, 3, "str", 4], [5, 6, 7, "str2", 9]])

print(arr.dtype)
print(arr[:, 0])
print(arr[0, :])

samples = np.array([100, 101, 102, 103, 104, 105, 106, 107])
result = samples[1:6:2]
print(result)  # 101,103,105

original = np.array([10, 20, 30, 40])
chunk = original[0:2]  # Grabbing [10, 20]
chunk[0] = 99

print(original.base)

prices_usd = np.array([10, 20, 30])

prices = np.multiply(prices_usd, 1.5)
print(prices)
print(prices * 0.2)


data = np.array([1, 2, 3, 4, 5, 6])
print(data.reshape(3, 2))

scores = np.array([55, 82, 90, 43, 76])
passed = scores >= 70
print(passed)

passing_scores = scores[passed]

print(scores)
print(passing_scores)

products = np.array([101, 999, 202, 999, 303])
locations = np.where(products == 999)
print(locations)

rain = np.array([0.0, 1.2, 0.5, 3.1, 0.0])
print(np.sum(rain))
print(np.max(rain))

# 100 random whole numbers between one and six
# np.random.rand(1, 6, 100)
np.random.randint(1, 7, size=100)
np.random.uniform(1, 6, 100)

# Rows = Students, Columns = Quizzes
scores = np.array(
    [
        [80, 90, 100],  # Student A
        [70, 75, 80],  # Student B
    ]
)
print(np.mean(scores, axis=1))
print(np.mean(scores, axis=0))
