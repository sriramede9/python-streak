# variables
name = "Alice"
age = 30
height = 5.8
isStudent = False
fee = 900.50

welcome_banner = """
Welcome to          the 
                            lazy world of 
            learning python
                                you should care or someone has the skills you always wanted!!!!                

"""

print("Name:",name, ",Age:", age, ",Height:", height,",Is Student:", isStudent)

print ("Age * Height =","-"*12, age ** 2 , "_"*25)

print(welcome_banner)
if not isStudent or not fee < 1000:
    print("Hello ",name)
else:
    print ("Hey I am Bart Simpson, who the hell are you??")

#  Let's do some exception handling
  
total = 500

try:
    subtotal = total/"abc"
except (ZeroDivisionError,TypeError) as z:
    print("do not divide by zero hero!!!") 
finally:
    print("we are over the try catch subtotal block")

numbers = [1,2,3,4,5]

try:
    print("well this is the lucky number",  numbers[5])
except (IndexError) as e:
    print("well ain't that lucky ",e)
finally:
    print("we are done with lucky numbers")

student = {
    "name":"Sri",
    "age" : 30,
    "isStudent":False
}            

try:
    open("student.txt")
    print("salary of :" , student["name"] , "is",student["salary"])
except (KeyError,FileNotFoundError) as e:
    print(f"invalid Key or file not found:{e}")
finally:
    print("hey Done with Student",student)        