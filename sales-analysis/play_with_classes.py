class Dog:
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed

class Cat:
    def __init__(self,name,breed):
        self.name = name
        self.breed = breed


mydog = Dog(name="D1",breed="b1")

print(mydog.name,mydog.breed)

class Animal:
    def __init__(self,name,breed):
        self.name = name
        self.breed = breed
        pass

    def eat(self):
        print(f"{self.name}, is eating")

    def sleep(self):
        print(f"{self.name}, is sleeping")
    
class Rhino(Animal):
    def run(self):
        print(f"Rhiono {self.name} can run")


mini_rhino = Rhino(name="R1",breed="The whity")

mini_rhino.eat()
mini_rhino.sleep()
mini_rhino.run()
