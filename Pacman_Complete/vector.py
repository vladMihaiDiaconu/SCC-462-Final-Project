import math

# Class to represent a 2-dimensional vector with x and y components
class Vector2(object):
    # Initializing a Vector2 object
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.thresh = 0.000001

    # Defining vector addition
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    # Defining vector subtraction
    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    # Defining vector negation
    def __neg__(self):
        return Vector2(-self.x, -self.y)

    # Defining scalar multiplication
    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)

    # Defining scalar division
    def __div__(self, scalar):
        if scalar != 0:
            return Vector2(self.x / float(scalar), self.y / float(scalar))
        return None

    # Defining true division
    def __truediv__(self, scalar):
        return self.__div__(scalar)

    # Defining vector equality comparison
    def __eq__(self, other):
        if abs(self.x - other.x) < self.thresh:
            if abs(self.y - other.y) < self.thresh:
                return True
        return False

    # Calculating the squared magnitude (length squared) of the vector
    def magnitudeSquared(self):
        return self.x**2 + self.y**2

    # Calculating the magnitude (length) of the vector
    def magnitude(self):
        return math.sqrt(self.magnitudeSquared())

    # Creating and returning a new Vector2 object with the same x and y values
    def copy(self):
        return Vector2(self.x, self.y)

    # Returning the vector's components as a tuple (x, y)
    def asTuple(self):
        return self.x, self.y

    # ReturnING the vector's components as a tuple of integers (int(x), int(y))
    def asInt(self):
        return int(self.x), int(self.y)

    # Defining the string representation of the vector object
    def __str__(self):
        return "<"+str(self.x)+", "+str(self.y)+">"