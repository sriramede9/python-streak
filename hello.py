# functions


def greet():
    print("Welcome Function")


greet()

# how to accept variables?


def greetPerson(name: str):
    print("Hey Buddy what's up", name.upper())
    print(f"Hey {name.upper()} what's ups")


greetPerson("Mikes")


def check_weather(data: int):
    if data > 25:
        print("Hot stay home")
    else:
        print("Go out you monster")


check_weather(18)


def check_weather2(data: int) -> str:
    if not isinstance(data, int):
        raise TypeError("data must be an integer")
    if data > 25:
        return "Hot stay home"
    else:
        return "Go out you monster"


try:
    decision = check_weather2("abc")
except TypeError as e:
    print(f"Type error {e}")


print(f"You better do this: {check_weather2(data=18)}")


def concatStrings(value1: str = "John", value2: str = "Doe") -> str:
    if not isinstance(value1, str) and not isinstance(value2, str):
        raise TypeError("Data must be str inngggggg")
    return value1 + value2


print("Ohoo", {concatStrings("Dve", "DJfs")})  # Hmm you are creating a set here dude
print("Ohoo", concatStrings(value1="Dve", value2="DJfs"))
print(f"Ohoo {concatStrings(value1='Dve', value2='DJfs')}")

# welcome to handle multiple return values behold java and pojo


def simple_function():
    nums = [1, 2, 3, 4, 5, 56, 0]
    first_value = nums[0]
    min_value = min(nums)
    max_value = max(nums)
    return first_value, min_value, max_value


a, b, c = simple_function()
print(a, b, c)

# import modules and part of modules

import math

math.sqrt(16)

# import only part of modules
from math import sqrt, pow 

pow(2, 3)

import random

random.randint(1, 100)
random.choice(["apple", "banana", "cherry"])

# date time
import datetime
today = datetime.date.today()
print(today)

import os

current_directory = os.getcwd()
print(current_directory)

import json
data = {"name": "John", "age": 30, "city": "New York"}
json_data = json.dumps(data)
print(json_data) # formatted JSON
print(data) # dictionary
print(type(json_data)) # Json data in string format
print(type(data)) # dictionary