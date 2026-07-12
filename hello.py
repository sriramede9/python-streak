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
  