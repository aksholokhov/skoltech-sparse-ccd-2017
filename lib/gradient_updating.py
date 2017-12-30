import numpy as np
import fibonacci_heap_mod as fhm
from scipy.sparse.linalg import norm

class GradientUpdateTool(object):
    """
    Base class for gradient maintaining routines in coordinate descent methods
    """

    def get_coordinate(self):
        raise Exception("Not implemented yet")

    def update(self, delta):
        raise Exception("Not implemented yet")

    def get(self):
        raise Exception("Not implemented yet")

    def get_norm(self):
        raise Exception("Not implemented yet")

class BasicGradientUpdater(GradientUpdateTool):

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

    def get_norm(self):
        return self.__g.norm()

class RidgeHeapGradientUpdater(GradientUpdateTool):

    def __init__(self, g):
        self.__g = g
        self.__g_norm = 0
        self.__heap = fhm.Fibonacci_heap()
        self.__g_elements = []

        if max(g.shape) <= 1e8:    # dense vectors work significantly better if not blow memory
            for i, val in enumerate(np.squeeze(g.toarray())):
                self.__g_elements.append(self.__heap.enqueue(i, val))
                self.__g_norm += val**2

        else:
            raise Exception("This method hasn't been tested yet on dimensions n > 10^8, sorry :(")

    def update(self, delta):
        g_elems = self.__g_elements
        heap = self.__heap

        delta_arr = np.squeeze(delta.toarray())

        for k in delta.nonzero()[1]:
            old_priority = g_elems[k].get_priority()
            new_priority = old_priority + delta_arr[k]
            if old_priority > new_priority:
                heap.decrease_key(entry=g_elems[k], new_priority=new_priority)

            else:
                value = g_elems[k].get_value()
                heap.decrease_key(entry=g_elems[k], new_priority=heap.min().get_priority() - 1)
                heap.dequeue_min()
                g_elems[k] = heap.enqueue(value=value, priority=new_priority)
            self.__g_norm = self.__g_norm - old_priority ** 2 + new_priority ** 2

        self.__g += delta

    def get(self):
        return self.__g

    def get_norm(self):
        return self.__g_norm

    def get_coordinate(self):
        return self.__heap.min().get_value()

class LassoHeapGradientUpdater(GradientUpdateTool):

    def __init__(self, g):
        self.__g = g
        self.__g_norm = 0
        self.__pos_heap = fhm.Fibonacci_heap()
        self.__neg_heap = fhm.Fibonacci_heap()
        self.__pos_g_elems = []
        self.__neg_g_elems = []

        if max(g.shape) <= 1e8:    # dense vectors work significantly better if not blow memory
            for i, val in enumerate(np.squeeze(g.toarray())):
                self.__pos_g_elems.append(self.__pos_heap.enqueue(i, val))
                self.__neg_g_elems.append(self.__neg_heap.enqueue(i, -val))
                self.__g_norm += val**2

        else:
            raise Exception("This method hasn't been tested yet on dimensions n > 10^8, sorry :(")

    def update(self, delta):
        pos_g_elems = self.__pos_g_elems
        neg_g_elems = self.__neg_g_elems
        pos_heap = self.__pos_heap
        neg_heap = self.__neg_heap

        for k in delta.nonzero()[1]:
            old_priority = pos_g_elems[k].get_priority()
            new_priority = old_priority + delta[0, k]
            if old_priority > new_priority:
                pos_heap.decrease_key(entry=pos_g_elems[k], new_priority=new_priority)

                value = neg_g_elems[k].get_value()
                neg_heap.decrease_key(entry=neg_g_elems[k], new_priority=neg_heap.min().get_priority() - 1)
                neg_heap.dequeue_min()
                neg_g_elems[k] = neg_heap.enqueue(value=value, priority= -new_priority)
            else:
                value = pos_g_elems[k].get_value()
                pos_heap.decrease_key(entry=pos_g_elems[k], new_priority=pos_heap.min().get_priority() - 1)
                pos_heap.dequeue_min()
                pos_g_elems[k] = pos_heap.enqueue(value=value, priority=new_priority)

                neg_heap.decrease_key(entry=neg_g_elems[k], new_priority= -new_priority)

            self.__g_norm = self.__g_norm - old_priority ** 2 + new_priority ** 2

        self.__g += delta

    def get(self):
        return self.__g

    def get_norm(self):
        return self.__g_norm

    def get_coordinate(self):
        a = self.__pos_heap.min().get_value()
        a_priority = self.__pos_heap.min().get_priority()
        b = self.__neg_heap.min().get_value()
        b_priority = self.__neg_heap.min().get_priority()

        return a if a_priority < b_priority else -b

class GSGradientUpdater(GradientUpdateTool):
    def __init__(self, g):
        self.__g = g

    def get_coordinate(self):
        return abs(self.__g).argmax()

    def get(self):
        return self.__g

    def update(self, delta):
        self.__g += delta

    def get_norm(self):
        return norm(self.__g)