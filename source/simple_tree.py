

class Node:
    def __init__(self, parent=None, successors=None):
        self.parent = parent
        self.successors = [] if successors is None else successors
        self.data = None

    def add_successor(self, s):
        self.successors.append(s)

    def insert_successor(self, s, i):
        self.successors.insert(i, s)

    def remove_successor(self, i):
        self.successors.pop(i)


class BinaryFilterNode(Node):
    def __init__(self, a, j, parent=None, successors=None):
        super().__init__(parent, successors)
        self.data = (a, j)

    def filter_data(self, data):
        left = None
        right = None
