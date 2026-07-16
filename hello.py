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
new_arr = np.array([1, 2, 3, 4, 5], ndmin=5)  # crates a 5D array [[[[[1,2,3,4,5]]]]]
# NumPy won't remove dimensions. It only adds singleton (size 1) dimensions if needed

# understand data types

int_arr = np.array([1, 2, 3, 4, 5, 6], dtype="i")
print(type(int_arr))

arr = np.array(["apple", "banana", "cherry"])

print(arr.dtype)

arr = np.array([1, 2, 3, 4], dtype="U")

print(arr)
print(arr.dtype)

try:
    arr = np.array(["a", "2", "3"], dtype="i")
except ValueError as e:
    print("Error:", e)

# Converting Data Type on Existing Arrays

double_arr = np.array([1.1, 2.2, 3.3, 4.4])
print(double_arr.dtype)  # float64
int_arr = double_arr.astype("i")  # convert to integer
print(int_arr)  # [1 2 3 4]

#  Diff between copy and view
#  copied data owns the copy, view doesn't it will modify original source like shallow copy

print(arr)

new_arr = arr.copy()
view_arr = arr.view()

new_arr[0] = "621"
view_arr[0] = "721"

print(arr)
print(new_arr)
print(view_arr)

#  shape
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(arr.shape)  # 2,1,4 means I have two boxes with 1X4 in each?
print(arr.ndim)  # 2 dim

streams_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

result = [e * 2 for e in streams_arr]  # prints each element
print(result)  # this will have scalar values in the ndarray
print(streams_arr * 2)  # doubles each element in the array

# the main diff between list and ndarray is that ndarray is more efficient in terms of memory and performance. It allows for vectorized operations, meaning you can perform operations on entire arrays without the need for explicit loops, which is not possible with standard Python lists.
# allso list stores in objects and ndarray stores in contiguous memory locations which makes it more efficient for numerical computations.
# plus , minus, multiplication, division and other operations are performed element-wise in ndarray, while in lists you would need to use loops or comprehensions to achieve similar results.


# understandin API's

from geopy.geocoders import Nominatim


def get_city_coordinates(city_name):
    # Initialize Nominatim API with a descriptive user agent (required by OSM policy)
    geolocator = Nominatim(user_agent="my_city_geocoder_app")

    try:
        # Request location data
        location = geolocator.geocode(city_name)

        if location:
            return {
                "address": location.address,
                "latitude": location.latitude,
                "longitude": location.longitude,
            }
        else:
            return "City not found."

    except Exception as e:
        return f"An error occurred: {e}"


import requests

# Build the API URL with our parameters


def get_weather_data(latitude, longitude):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException as e:
        return {"error": f"An error occurred while fetching weather data: {e}"}
    else:
        if response.status_code == 200:
            return response.json()
        else:
            print(
                f"Error: Received status code {response.status_code} from the weather API."
            )
            return {"error": "Unable to fetch weather data."}


# Example Usage
city = "Hudson Bay, Canada"
coordinates = get_city_coordinates(city)
print(coordinates)

# We need coordinates to get weather data
latitude = coordinates["latitude"]  #  latitude
longitude = coordinates["longitude"]  #  longitude

get_weather = get_weather_data(latitude, longitude)  # to check exception handling
get_weather = get_weather_data(95, 45)  # to check exception handling
print(get_weather)
