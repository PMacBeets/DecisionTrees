import math
from collections import deque

import pandas as pd
import numpy as np

import ID3_Plus
from g import *


class Node:
    """Contains the information of the node and another nodes of the Decision Tree."""

    def __init__(self,ss):
        self.ss = ss
        self.uid = str(g.uid)
        g.uid+=1
        self.name = None
        self.value = None
        self.next = None
        self.childs = None
        self.parent = None
        self.id = None
        self.x_ids = None
        self.visited = []
        self.obsv = []
        self.unvisited = ss.labels
        self.df = None
        self.cost = None


    def create_df(self):
        #print(f"{self.x_ids},{self.unvisited}")
        data = self.ss.X[self.x_ids,:]
        data = data[:,self.unvisited]
        self.df = pd.DataFrame(data=data,columns=np.array(self.ss.feature_names)[self.unvisited],index=np.array(self.ss.labels)[self.x_ids]) #,,

    def calc_prob_obs(self):
        pass

class DecisionTreeClassifier:
    """Decision Tree Classifier using ID3 algorithm."""

    def __init__(self, search_state, debug=False):
        self.ss = search_state
        self.node = None
        #self.entropy = self._get_entropy([x for x in range(len(self.ss.labels))],debug)  # calculates the initial entropy

    def _get_entropy(self, x_ids,debug):
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
        label_count_new = self.ss.probY[x_ids]
        #label_count = [labels.count(x) for x in self.ss.labelCategories]
        # calculate the entropy for each category and sum them

        # entropy = 0
        # for count in label_count:
        #     if count:
        #         entropy -= count / len(x_ids) * math.log(count / len(x_ids), 2)
        #         print(f"        entropy = {entropy}")

        entropy_new = 0
        #label_count_new = label_count_new/np.sum(label_count_new)
        for count in label_count_new:
            if count:
                entropy_new -= count * math.log(count, 2)
                if debug: print(f"        entropy new = {entropy_new}")

        if debug:
            # print(f"    {label_count}")
            # print(f"    entropy = {entropy}")
            print(f"    {label_count_new}")
            print(f"    entropy new = {entropy_new}")
        return entropy_new

    def _get_information_gain(self, x_ids, feature_id, last_id,node,debug):
        """Calculates the information gain for a given feature based on its entropy and the total entropy of the system.
        Parameters
        __________
        :param x_ids: list, List containing the instances ID's
        :param feature_id: int, feature ID
        __________
        :return: info_gain: float, the information gain for a given feature.
        """
        # calculate total entropy
        if debug: print(f"Get Total Entropy")
        info_gain = self._get_entropy(x_ids,debug)
        if debug:  print(f"System Entropy = {info_gain}")
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

        cost = 0
        if last_id is None:
            # The first options is the average distance
            # print(f"last ID is None = {last_id}")
            cost = self.ss.start[feature_id]+self.ss.sensor_cost[feature_id]
            if debug: print(f"From Start -> {self.ss.feature_names[feature_id]} ")
        else:
            cost = self.ss.connection_matrix[last_id, feature_id]+self.ss.sensor_cost[feature_id]
            if debug: print(
                f"From {self.ss.feature_names[last_id]} ({node.obsv[-1]}) -> {self.ss.feature_names[feature_id]} ")

        # compute the information gain with the chosen feature
        for val_counts, val_ids in zip(feature_vals_count, feature_vals_id):
            entropy = np.sum(self.ss.probY[val_ids]) * self._get_entropy(val_ids, debug)
            if debug: print(f"  {info_gain - entropy} = {info_gain} - {entropy}")
            info_gain = info_gain - entropy
            # a = 1
            # for i in range(len(self.ss.feature_names)):
            #     # Iterate through events
            #     for j in range(len(self.ss.labels)):
            #         a = a*self.ss.k_dic[j][self.ss.X[i][j]]
        # for val, val_ids in zip(feature_vals, feature_vals_id):
        #     entropy  = self._get_conditional_entropy(val_ids, node,feature_id, debug)
        #     if debug: print(f"  {info_gain-entropy} = {info_gain} - {entropy}")
        #     info_gain = info_gain - entropy
                                      #*val_counts / len(x_ids) *
        #print(f"last ID = {last_id}")

        if debug:

            print(
                f"Cost = {cost:.2f} Info = {info_gain:.2f}, Info/Cost = {info_gain/cost}")
        return info_gain,cost

    def _get_feature_max_information_gain(self, x_ids, feature_ids, last_id,node, debug):
        """Finds the attribute/feature that maximizes the information gain.
        Parameters
        __________
        :param x_ids: list, List containing the samples ID's
        :param feature_ids: list, List containing the feature ID's
        __________
        :returns: string and int, feature and feature id of the feature that maximizes the information gain
        """
        # get the entropy for each feature
        #print(f"last ID = {last_id}")
        cost_arr = np.zeros(len(feature_ids))
        features_entropy_arr = np.zeros(len(feature_ids))
        i=0
        for feature_id in feature_ids:
            features_entropy,cost = self._get_information_gain(x_ids, feature_id, last_id,node, debug)
            cost_arr[i] = cost
            features_entropy_arr[i] = features_entropy
            i+=1
        # find the feature that maximises the information gain
        index = np.argmax(features_entropy_arr/cost_arr)
        max_id = feature_ids[index]
        return self.ss.feature_names[max_id], max_id, cost_arr[index]

    def id3(self, max_depth=None, debug=None):
        """Initializes ID3 algorithm to build a Decision Tree Classifier.
        :return: None
        """
        x_ids = [x for x in range(len(self.ss.X))]
        feature_ids = [x for x in range(len(self.ss.feature_names))]
        self.node = self._id3_recv(x_ids, feature_ids, self.node, 0, max_depth, [], list(np.arange(len(self.ss.feature_names))),None, None, debug)
        print('')

    def _id3_recv(self, x_ids, feature_ids, node, depth, max_depth,visited, unvisited,obsv, last_id, debug):
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

            node.unvisited = unvisited.copy()
            if visited:
                #print(visited)
                node.visited = visited.copy()
                node.obsv = obsv.copy()

        # sorted labels by instance id
        labels_in_features = [self.ss.labels[x] for x in x_ids]
        # if all the example have the same class (pure node), return node
        if len(set(labels_in_features)) == 1:
            node.value = self.ss.labels[x_ids[0]]
            if debug: print(f"Failure Found: {node.value}")
            return node
        # if there are not more feature to compute, return node with the most probable class
        if len(feature_ids) == 0:
            print(f"Feature ID {feature_ids}")
            node.value = max(set(labels_in_features), key=labels_in_features.count)  # compute mode
            if debug: print(f"No More Features: {node.value} chosen from {labels_in_features}")
            return node
        # If the maximum depth is reached return the set of all possible outcomes
        if max_depth is not None and depth == max_depth:
            node.value = set(labels_in_features) # compute mode
            return node
        # else...
        # choose the feature that maximizes the information gain
        if debug: print(f"last ID = {last_id}")
        best_feature_name, best_feature_id, cost = self._get_feature_max_information_gain(x_ids, feature_ids, last_id,node, debug)
        node.value = best_feature_name
        node.visited.append(best_feature_id)
        node.unvisited.remove(best_feature_id)
        node.name = best_feature_name
        node.x_ids = x_ids
        node.id = best_feature_id
        node.childs = []
        node.cost = cost
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
            child.parent = node
            child.visited = node.visited.copy()
            child.unvisited = node.unvisited.copy()
            child.obsv = node.obsv.copy()
            child.obsv.append(value)
            if not child_x_ids:
                child.next = max(set(labels_in_features), key=labels_in_features.count)
                print('')
            else:
                feature_ids_copy = feature_ids.copy()
                if feature_ids_copy and best_feature_id in feature_ids:
                    to_remove = feature_ids_copy.index(best_feature_id)
                    if debug: print(f"Remove from feature ID's {to_remove}")
                    feature_ids_copy.pop(to_remove)
                # recursively call the algorithm
                child.next = self._id3_recv(child_x_ids, feature_ids_copy, child.next, depth+1, max_depth, node.visited, node.unvisited,child.obsv, node.id, debug)
        #if debug: print(f"END: {node.value} chosen from {labels_in_features}")
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