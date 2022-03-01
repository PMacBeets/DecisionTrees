import math
from collections import deque

import pandas as pd
import numpy as np

import ID3_Plus


class Node:
    """Contains the information of the node and another nodes of the Decision Tree."""

    def __init__(self,ss):
        self.ss = ss
        self.name = None
        self.value = None
        self.next = None
        self.childs = None
        self.id = None
        self.x_ids = None
        self.visited = []
        self.unvisited = ss.labels
        self.df = None

    def create_df(self):
        #print(f"{self.x_ids},{self.unvisited}")
        data = self.ss.X[self.x_ids,:]
        data = data[:,self.unvisited]
        self.df = pd.DataFrame(data=data,columns=np.array(self.ss.feature_names)[self.unvisited],index=np.array(self.ss.labels)[self.x_ids]) #,,


class DecisionTreeClassifier:
    """Decision Tree Classifier using ID3 algorithm."""

    def __init__(self, search_state):
        self.ss = search_state
        self.node = None
        self.entropy = self._get_entropy([x for x in range(len(self.ss.labels))])  # calculates the initial entropy

    def _get_entropy(self, x_ids):
        """ Calculates the entropy.
        Parameters
        __________
        :param x_ids: list, List containing the instances ID's
        __________
        :return: entropy: float, Entropy.
        """
        # sorted labels by instance id
        labels = [self.ss.labels[i] for i in x_ids]
        # count number of instances of each category
        label_count_new = self.ss.probY
        #label_count = [labels.count(x) for x in self.labelCategories]
        # calculate the entropy for each category and sum them
        #entropy = sum([-count / len(x_ids) * math.log(count / len(x_ids), 2) if count else 0 for count in label_count])
        entropy_new = sum([-count * math.log(count, 2) if count else 0 for count in label_count_new])
        return entropy_new

    def _get_information_gain(self, x_ids, feature_id, node):
        """Calculates the information gain for a given feature based on its entropy and the total entropy of the system.
        Parameters
        __________
        :param x_ids: list, List containing the instances ID's
        :param feature_id: int, feature ID
        __________
        :return: info_gain: float, the information gain for a given feature.
        """
        # calculate total entropy
        info_gain = self._get_entropy(x_ids)
        # store in a list all the values of the chosen feature
        x_features = [self.ss.X[x][feature_id] for x in x_ids]
        # get unique values
        feature_vals = list(set(x_features))
        # get frequency of each value
        feature_vals_count = [x_features.count(x) for x in feature_vals]
        # get the feature values ids
        feature_vals_id = [
            [x_ids[i]
            for i, x in enumerate(x_features)
            if x == y]
            for y in feature_vals
        ]

        # compute the information gain with the chosen feature
        info_gain = info_gain - sum([val_counts / len(x_ids) * self._get_entropy(val_ids)
                                     for val_counts, val_ids in zip(feature_vals_count, feature_vals_id)])

        return info_gain/self.ss.sensor_cost[feature_id] # Added by Jake, Information per cost

    def _get_feature_max_information_gain(self, x_ids, feature_ids, node):
        """Finds the attribute/feature that maximizes the information gain.
        Parameters
        __________
        :param x_ids: list, List containing the samples ID's
        :param feature_ids: list, List containing the feature ID's
        __________
        :returns: string and int, feature and feature id of the feature that maximizes the information gain
        """
        # get the entropy for each feature
        features_entropy = [self._get_information_gain(x_ids, feature_id, node) for feature_id in feature_ids]
        # find the feature that maximises the information gain
        max_id = feature_ids[features_entropy.index(max(features_entropy))]

        return self.ss.feature_names[max_id], max_id

    def id3(self, max_depth=None):
        """Initializes ID3 algorithm to build a Decision Tree Classifier.
        :return: None
        """
        x_ids = [x for x in range(len(self.ss.X))]
        feature_ids = [x for x in range(len(self.ss.feature_names))]
        self.node = self._id3_recv(x_ids, feature_ids, self.node,0,max_depth,[], list(np.arange(len(self.ss.feature_names))))
        print('')

    def _id3_recv(self, x_ids, feature_ids, node, depth, max_depth,visited, unvisited):
        """ID3 algorithm. It is called recursively until some criteria is met.
        Parameters
        __________
        :param x_ids: list, list containing the samples ID's
        :param feature_ids: list, List containing the feature ID's
        :param node: object, An instance of the class Nodes
        __________
        :returns: An instance of the class Node containing all the information of the nodes in the Decision Tree
        """

        if not node:
            node = Node(self.ss)  # initialize nodes
            node.name = None
            node.visited = visited.copy()
            node.unvisited = unvisited.copy()

        # sorted labels by instance id
        labels_in_features = [self.ss.labels[x] for x in x_ids]
        # if all the example have the same class (pure node), return node
        if len(set(labels_in_features)) == 1:
            node.value = self.ss.labels[x_ids[0]]
            return node
        # if there are not more feature to compute, return node with the most probable class
        if len(feature_ids) == 0:
            node.value = max(set(labels_in_features), key=labels_in_features.count)  # compute mode
            return node

        # If the maximum depth is reached return the set of all possible outcomes
        if max_depth is not None and depth == max_depth:
            node.value = set(labels_in_features) # compute mode
            return node
        # else...
        # choose the feature that maximizes the information gain
        best_feature_name, best_feature_id = self._get_feature_max_information_gain(x_ids, feature_ids, node)
        node.value = best_feature_name
        node.visited.append(best_feature_id)
        node.unvisited.remove(best_feature_id)
        node.name = best_feature_name
        node.x_ids = x_ids
        node.id = best_feature_id
        node.childs = []
        # value of the chosen feature for each instance
        feature_values = list(set([self.ss.X[x][best_feature_id] for x in x_ids]))

        # loop through all the values
        for value in feature_values:
            child = Node(self.ss)
            child.name = best_feature_name+" (" + str(value) +")"
            child.value = value  # add a branch from the node to each feature value in our feature
            node.id = best_feature_id
            node.childs.append(child)  # append new child node to current node
            child_x_ids = [x for x in x_ids if self.ss.X[x][best_feature_id] == value]
            child.x_ids = child_x_ids
            child.visited = node.visited.copy()
            child.unvisited = node.unvisited.copy()
            if not child_x_ids:
                child.next = max(set(labels_in_features), key=labels_in_features.count)
                print('')
            else:
                if feature_ids and best_feature_id in feature_ids:
                    to_remove = feature_ids.index(best_feature_id)
                    feature_ids.pop(to_remove)
                # recursively call the algorithm
                child.next = self._id3_recv(child_x_ids, feature_ids, child.next, depth+1, max_depth, node.visited, node.unvisited)
        return node

    def printTree(self):
        if not self.node:
            return
        nodes = deque()
        nodes.append(self.node)
        while len(nodes) > 0:
            node = nodes.popleft()
            print(node.value)
            if node.childs:
                for child in node.childs:
                    print('({})'.format(child.value))
                    nodes.append(child.next)
            elif node.next:
                print(node.next)