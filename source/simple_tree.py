

class Node:
    def __init__(self, parent=None, successors=None):
        self.parent = parent
        self.successors = [] if successors is None else successors
        self.data = None

    def add_successor(self, s):
        self.successors.append(s)
        s.parent = self

    def insert_successor(self, s, i):
        self.successors.insert(i, s)
        s.parent = self

    def remove_successor(self, i):
        s = self.successors.pop(i)
        s.parent = None


class BinaryFilterNode(Node):
    def __init__(self, a=None, j=None, parent=None, successors=None):
        super().__init__(parent, successors)
        self.data = (a, j)

    def filter_data(self, data):
        left_cond = data[0][self.data[1]] <= self.data[0]
        right_cond = data[0][self.data[1]] > self.data[0]
        left = (data[0][left_cond], data[1][left_cond])
        right = (data[0][right_cond], data[1][right_cond])
        return left, right


class LeafNode(Node):
    def __init__(self, value, parent=None, successors=None):
        super().__init__(parent, successors)
        self.value = value
