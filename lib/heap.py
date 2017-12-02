class Node(object):
    def __init__(self, k, v):
        self.__k = k
        self.__v = v
        self.__left = None
        self.__right = None
        self.__parent = None

    @classmethod
    def fromOther(cls, other):
        nd = Node(other.get_value(), other.get_priority())
        nd.__left = other.__left
        nd.__right = other.__right
        nd.__parent = other.__parent
        return nd

    def get_value(self):
        return self.__k

    def get_priority(self):
        return self.__v


class myheap(object):

    def __init__(self, nodes):
        self.__root = None

    def min(self):
        return self.__root

    def enqueue(self, k, v):
        entry = Node(k, v)
        if self.__root is None:
            self.__root = entry
        current = self.__root
        while True:
            if current.__v <= entry.v:
                if current.__left == None:
                    current.__left = entry
                    break
                else:
                    current = current.__left
            else:
                if current.__right == None:
                    current.__right = entry
                    break
                else:
                    current = current.__right
        return entry

    def increaseKey(self, entry, new_priority):
        entry.__v = new_priority
        current = entry
        while True:
            if entry.__parent.get_priority() >= entry.get_priority():
                swap(entry.__parent, entry)
                current = entry.__parent
                continue
            break
        return current


    def decreaseKey(self, entry,):


def swap(self, a, b):
    t = Node.fromOther(a)
    a.__k = b.__k
    a.__v = b.__v
    a.__left = b.__left
    a.__right = b.__right
    a.__parent = b.__parent
    b.__k = t.__k
    b.__v = t.__v
    b.__left = t.__left
    b.__right = t.__right
    b.__parent = t.__parent

