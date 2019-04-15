import numpy as np
import matplotlib.pyplot as plt
import time
from simple_tree import *
from matplotlib.colors import ListedColormap

# caret R language (ml package)
# meltdown and specter
# Titanic machine learning kaggle
# light gbm


def gini(data_y, frequencies):
    return np.sum(frequencies[data_y] * (1 - frequencies[data_y]))


class DecisionTreeClassifier:
    def __init__(self):
        self.data = None
        self.data_index = None
        self.frequencies = None
        self.classes = None
        self.class_indices = None
        self.n_classes = None
        self.n_features = None
        self.n_samples = None
        self.root = None

    def fit(self, data, min_count=1, max_depth=1e100):
        self.data = data
        self.n_features = len(data[0][0])
        self.n_samples = len(data[1])
        self.classes = np.unique(data[1], return_counts=True)
        self.n_classes = len(self.classes[0])
        self.class_indices = {v: i for i, v in enumerate(self.classes[0])}
        self.frequencies = self.classes[1] / np.sum(self.classes[1])
        self.data_index = (data[0], np.array([self.class_indices[y] for y in data[1]]))
        self.root = BinaryFilterNode()
        self.fit_gini([i for i in range(self.n_samples)], self.root, min_count=min_count, max_depth=max_depth)

    def fit_gini(self, pos, node, min_count=1, max_depth=1e10, depth=1):
        m = len(pos)
        min_j = None
        min_i = None
        min_value = 1e+300
        for j in range(self.n_features):
            pos_j = sorted(pos, key=lambda x: self.data_index[0][x, j])
            for i in range(m - 1):
                left_pos = pos_j[:i + 1]
                right_pos = pos_j[i + 1:]
                left_data = self.data_index[1][left_pos]
                right_data = self.data_index[1][right_pos]
                left_frequencies = np.bincount(left_data, minlength=self.n_classes) / (i + 1)
                right_frequencies = np.bincount(right_data, minlength=self.n_classes) / (m - (i + 1))
                value = ((i + 1) / m * gini(left_data, left_frequencies) +
                        (m - i - 1) / m * gini(right_data, right_frequencies))
                if value <= min_value:
                    min_value = value
                    min_j = j
                    min_i = i
        pos_j = sorted(pos, key=lambda x: self.data_index[0][x, min_j])
        left_pos = list(pos_j[:min_i + 1])
        right_pos = list(pos_j[min_i + 1:])
        left_data = self.data_index[1][left_pos]
        right_data = self.data_index[1][right_pos]
        node.data = (self.data_index[0][pos_j[min_i], min_j], min_j)
        left_node = BinaryFilterNode()
        right_node = BinaryFilterNode()
        if min_i + 1 <= min_count or depth == max_depth:
            node.add_successor(
                LeafNode([len(left_data[left_data == i]) / len(left_pos) for i in range(self.n_classes)]))
        else:
            # print("l:", left_pos)
            node.add_successor(left_node)
            self.fit_gini(left_pos, left_node, min_count=min_count, max_depth=max_depth, depth=depth + 1)
        if m - (min_i + 1) <= min_count or depth == max_depth:
            node.add_successor(
                LeafNode([len(right_data[right_data == i]) / len(right_pos) for i in range(self.n_classes)]))
        else:
            # print("r:", right_pos)
            node.add_successor(right_node)
            self.fit_gini(right_pos, right_node, min_count=min_count, max_depth=max_depth, depth=depth + 1)

    def predict(self, x):
        node = self.root
        while not isinstance(node, LeafNode):
            a = node.data[0]
            j = node.data[1]
            if x[j] <= a:
                node = node.successors[0]
            else:
                node = node.successors[1]
        return node.value


def get_meshgrid(data, step=0.01, border=2.0):
    min_x, max_x = np.min(data[0][:, 0]) - border, np.max(data[0][:, 0]) + border
    min_y, max_y = np.min(data[0][:, 1]) - border, np.max(data[0][:, 1]) + border
    return np.meshgrid(np.arange(min_x, max_x, step), np.arange(min_y, max_y, step))


if __name__ == "__main__":
    from sklearn import datasets, metrics
    from sklearn.model_selection import train_test_split

    d = datasets.make_classification(10000, n_features=2, n_informative=2, n_redundant=0, n_classes=3,
                                     n_clusters_per_class=1, random_state=4)
    x_train, x_test, y_train, y_test = train_test_split(d[0], d[1], test_size=0.3, random_state=4)

    start_time = time.time()
    dtc = DecisionTreeClassifier()
    dtc.fit((x_train, y_train), min_count=1, max_depth=5)
    end_time = time.time() - start_time
    print("Decision tree fit. Elapsed:", end_time)

    start_time = time.time()
    p_test = np.array([dtc.classes[0][np.argmax(dtc.predict(x))] for x in x_test])
    acc_test = metrics.accuracy_score(y_test, p_test)
    end_time = time.time() - start_time
    print("Predict. Elapsed:", end_time)
    print("Accuracy test:", acc_test)

    start_time = time.time()
    p_train = np.array([dtc.classes[0][np.argmax(dtc.predict(x))] for x in x_train])
    acc_train = metrics.accuracy_score(y_train, p_train)
    end_time = time.time() - start_time
    print("Predict. Elapsed:", end_time)
    print("Accuracy train:", acc_train)

    start_time = time.time()
    xx, yy = get_meshgrid(d)
    p_grid = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            p_grid[i, j] = dtc.classes[0][np.argmax(dtc.predict([xx[i, j], yy[i, j]]))]
    end_time = time.time() - start_time
    print("Mesh generation. Elapsed:", end_time)
    colors = ListedColormap(["red", "blue", "yellow"])
    colors_light = ListedColormap(['lightcoral', 'lightblue', 'lightyellow'])
    fig, ax = plt.subplots()
    ax.pcolormesh(xx, yy, p_grid, cmap=colors_light)
    ax.scatter(list(map(lambda x: x[0], d[0])), list(map(lambda x: x[1], d[0])),
               c=d[1], cmap=colors, s=100)
    plt.show()
