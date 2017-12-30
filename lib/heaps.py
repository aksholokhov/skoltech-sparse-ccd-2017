from scipy import sparse
import numpy as np
import fibonacci_heap_mod as fhm

class Heap(object):
    """
    Yet another default heap interface
    """

    def get_min(self):
        raise Exception("Not implemented yet")

    def insert(self, k, v):
        raise Exception("Not implemented yet")

    def update(self, k, v):
        raise Exception("Not implemented yet")


class BinaryHeap(Heap):

    def __init__(self, data):
        N = max(data.shape)
        self.__N = N
        self.__size = 0
        self.__idx = -1*np.ones(N)
        self.__k = -1*np.ones(N)
        self.__v = -1*np.ones(N)

        if sparse.issparse(data):
            data = np.squeeze(data.toarray())

        for k, v in enumerate(data):
            if v != 0:
                self.insert(k, v)


    def __swap(self, i, j):
        a = self.__v[i]
        self.__v[i] = self.__v[j]
        self.__v[j] = a

        self.__idx[self.__k[i]] = j
        self.__idx[self.__k[j]] = i

        a = self.__k[i]
        self.__k[i] = self.__k[j]
        self.__k[j] = a


    def __siftUp(self, i):
        v = self.__v
        while v[i] < v[int((i-1)/2)]:
            self.__swap(i, int((i-1)/2))
            i = int((i-1)/2)


    def __siftDown(self, i):
        v = self.__v
        while 2 * i + 1 < self.__size:
            left = 2 * i + 1
            right = 2 * i + 2
            j = left
            if right < self.__size and v[right] < v[left]:
                j = right
            if v[i] <= v[j]:
                break
            self.__swap(i, j)
            i = j


    def update(self, k, v):
        pos = self.__idx[k]
        if self.__v[pos] > v:
            self.__v[pos] = v
            self.__siftUp(pos)
        else:
            self.__v[pos] = v
            self.__siftDown(pos)


    def insert(self, k, v):
        self.__size += 1
        size = self.__size
        self.__v[size-1] = v
        self.__k[size-1] = k
        self.__idx[k] = size-1


    def get_min(self):
        if self.__v[0] < 0:
            return self.__k[0]
        return -1


class fibonnacci_heap(Heap):

    def __init__(self, data):
        self.__heap = fhm.Fibonacci_heap()
        self.__elements = []

        if sparse.issparse(data):
            data = np.squeeze(data.toarray())

        for k, v in enumerate(data):
            self.insert(k, v)


    def get_min(self):
        self.__heap.min().get_value()


    def insert(self, k, v):
        self.__elements.append(self.__heap.enqueue(k, v))


    def update(self, k, v):
        elements = self.__elements
        heap = self.__heap

        if elements[k].get_priorty() > v:
            heap.decrease_key(entry=elements[k], new_priority=v)
        else:
            value = elements[k].get_value()
            heap.decrease_key(entry=elements[k], new_priority=heap.min().get_priority() - 1)
            heap.dequeue_min()
            elements[k] = heap.enqueue(value=value, priority=v)
