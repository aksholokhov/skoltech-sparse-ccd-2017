import numpy as np
import fibonacci_heap_mod as fhm

def GradientUpdateTool(Object):
    """
    Base class for gradient maintaining routines in coordinate descent methods
    """

    def get_coordinate(self):
        raise Exception("Not implemented yet")

    def update(self, delta):
        raise Exception("Not implemented yet")

    def get(self):
        raise Exception("Not implemented yet")

def BasicGradientUpdater(GradientUpdateTool):

    def __init__(self, g, type = "min"):
        self.__g = g

    def get_coordinate(self):
        if type is "min":
            return self.__g.argmin()
        elif type is "max":
            return self.__g.argmax()
        else:
            raise Exception("Not implemented yet")

    def update(self, delta):
        self.__g += delta

    def get(self):
        return self.__g

def HeapGradientUpdater(GradientUpdateTool):

    def __init__(self, g, type = "min"):
