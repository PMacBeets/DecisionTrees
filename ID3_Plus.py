import numpy as np
import pandas as pd
import math
from ID3 import DecisionTreeClassifier


class Node:

    def __init__(self, data):
        self.children = None #will contain a dictionary of pointers to child nodes
        self.data = data #Possible Events at this stage of the decision tree
        # Link state structure to given node
        self.data.node = self

class InspectionSystem:
    def __init__(self, path=None,debug=False):
        self.path = path
        self.inspection_df = None # Inspection measurements per
        self.sensor_df = None # Used for SD
        self.data_df = None # inspection_df + sensor_df
        self.event_prob = None # The probability of each event occurring
        self.connection_matrix = None # What is the cost of transitioning between connection matrices
        self.event_list = None # Label associated with each event (row of dataframe)

        # How many
        self.num_sensors = None
        self.num_inspections = None
        self.num_failure = None

        if path is None:
            self.build_random_data(debug)
        else:
            # Read in data from tables
            pass

        self.ave_inspection_costs = self.calc_ave_inspection_cost() # from connection matrix
        self.root = SearchState(np.array(self.inspection_df), list(self.inspection_df.columns), self.event_list, self.ave_inspection_costs, self.event_prob,self.connection_matrix)

    def lookahead(self):
        # instantiate DecisionTreeClassifier
        self.tree_clf = DecisionTreeClassifier(self.root)
        print("System entropy {:.4f}".format(self.tree_clf.entropy))
        # run algorithm id3 to build a tree
        self.tree_clf.id3(1)
        self.tree_clf.printTree()

    def buildtree(self, debug):
        # instantiate DecisionTreeClassifier
        self.tree_clf = DecisionTreeClassifier(self.root)
        #print("System entropy {:.4f}".format(self.tree_clf.entropy))
        # run algorithm id3 to build a tree
        self.tree_clf.id3(debug=debug)
        self.tree_clf.printTree()

    def search(self):
        root = Node(self.root)
        node_queue = [root]
        self.recursive_function(node_queue, debug=True)

    def recursive_function(self, queue, debug: bool):
        pass
        # pop node from queue if queue has nodes to look at
        if queue:
            node = queue.pop(0)
        else:
            print("End Of Search")
            return

            # Given "get_system info" Determine Sensor (Done)
        # Outcome_Table = outcome_table(node.data,debug=True)

        if debug: node.data.selfprint()
        # condition table
        #node.create_condition_table(node.data, debug=False)
        index = node.data.Information_Gain(node, "2", debug=True, average=True)
        # Create nodes for each child branch and copy/adjust "get_system info" fields if applicable
        node.data.check_children(node, index, queue, debug=True)
        self.recursive_function(queue, debug=debug)

    def build_random_data(self, debug=False):
        np.random.seed(1)  # 1 give unique values
        self.num_sensors = 3
        self.num_inspections = 9
        self.num_failure = 15

        data = {
            'Sensor': [-1, 0, 1],
            'Inspection': [1, 0],
        }

        self.inspection_df = pd.DataFrame()
        self.sensor_df = pd.DataFrame()

        self.event_list = []

        for i in range(self.num_failure):
            for j in range(self.num_sensors):
                self.sensor_df.loc[i, 'Sensor' + str(j)] = np.random.choice(data['Sensor'], 1)[0]
            for j in range(self.num_inspections):
                self.inspection_df.loc[i, 'Inspection' + str(j)] = np.random.choice(data['Inspection'], 1)[0]

            self.event_list.append("Failure " + str(i))

        # Add Nominal columns
        self.event_list.append("Nominal")
        for j in range(self.num_sensors):
            self.sensor_df.loc[self.num_failure + 1, 'Sensor' + str(j)] = 0
        for j in range(self.num_inspections):
            self.inspection_df.loc[self.num_failure + 1, 'Inspection' + str(j)] = 0

        self.data_df = pd.concat([self.sensor_df, self.inspection_df], axis=1)

        #self.data_df.head()

        (new_df, arr) = self.get_unique(self.data_df)
        print(f"Total failure coverage is {new_df.shape[0] / len(arr) * 100}%")
        (new_sensors_df, sensor_arr) = self.get_unique(self.sensor_df)
        print(f"Total sensor coverage is {new_sensors_df.shape[0] / len(sensor_arr) * 100}%")
        (new_inspection_df, inspection_arr) = self.get_unique(self.inspection_df)
        print(f"Total symptom coverage is {new_inspection_df.shape[0] / len(inspection_arr) * 100}%")

        # now we have the propagation table and the data
        # Now provide the probability of each failure
        # This in effect can be supplied by SD
        rel_probability = np.random.rand(self.num_failure+1)
        self.event_prob = rel_probability / np.sum(rel_probability)

        # Create connection matrix
        p = 0.3  # probability that there is a connection between two nodes
        self.connection_matrix = np.empty((self.num_inspections, self.num_inspections),dtype=float)
        self.connection_matrix[:] = 1000
        random = np.random.rand(self.num_inspections, self.num_inspections)
        # fill connection connection_matrix
        for i in range(self.num_inspections):
            for j in range(self.num_inspections):
                if i != j and random[i, j] > p:
                    self.connection_matrix[i, j] = float(10.0 * random[i, j])

    def calc_ave_inspection_cost(self):
        """
        Given the connection matrix calculate the average cost of getting to each inspection assuming the chance of
        being at each node is identical
        :return:
        """
        # Row = Current location
        # Column = next location
        ave_inspection_costs = np.zeros(self.num_inspections)

        for i in range(self.num_inspections):
            filter = ~np.isnan(self.connection_matrix[:,i])
            ave_inspection_costs[i]= np.average(self.connection_matrix[filter,i])

        return ave_inspection_costs


    def match_np_in_list(self,lst, comp):
        i = 0
        for arr in lst:
            if np.all(arr == comp):
                return i
            i += 1

        return -1

    def get_unique(self,data_frame):
        """
        Error is stopped
        :return:
        """
        unique_list = []

        # print(array_np)
        arr = np.zeros(data_frame.shape[0])
        j = 0
        for i in range(data_frame.shape[0]):
            res = self.match_np_in_list(unique_list, data_frame.iloc[i, :])
            if res == -1:  # res is false and no match has been found
                unique_list.append(data_frame.iloc[i, :])
                arr[i] = j
                j += 1
            else:
                arr[i] = res

        new_df = pd.DataFrame(unique_list, columns=data_frame.columns)
        return (new_df, arr)


class SearchState:

    def __init__(self, X, feature_names, event_list, sensor_cost, ProbY, connection_matrix,debug=False):
        assert isinstance(X, np.ndarray)
        #assert isinstance(event_list, list)

        self.X = np.array(X)
        self.feature_names = feature_names # inspection names
        self.labels = event_list
        self.probY = ProbY
        self.sensor_cost = sensor_cost # The cost of inspecting each sensor
        self.k_dic = [None]*np.shape(self.X)[1]
        self.labelCategories = list(set(self.labels))
        self.labelCategoriesCount = [list(self.labels).count(x) for x in self.labelCategories]
        self.connection_matrix = connection_matrix
        self.create_k_dict(debug)
        #self.create_condition_table(debug)

        # Sensor to be added to the sensor set on the next branch
        self.tree_clf = None

    def check_prop_is_valid(self,search_state):
        if abs(np.sum(search_state.probY) - 1) > 0.001:
            print(search_state.probY)
            print("Sum is: ", np.sum(search_state.probY))
            raise Exception("Probability of all outcomes must sum to 1")

    def splice_system_info(self, index, remove_events, k, debug: bool):
        # Remove events that do not correspond to the meaured output
        # iterate through event
        for i in range(len(self.labels)):
            if self.X[i][index] != k:
                remove_events.append(i)

        if debug:
            print("Remove Events: ", remove_events)
            print("Remove Sensor: ", index)

        # Do the following for Proptable,ProbY,ProbX,sensor_list,event_list
        # remove previous sensor
        X = np.delete(self.X, index, axis=1)
        sensor_list = np.delete(self.feature_names, index)
        sensor_cost = np.delete(self.sensor_cost, index)

        # remove events
        X = np.delete(X, remove_events, axis=0)
        probY = np.delete(self.probY, remove_events, axis=0)
        event_list = np.delete(self.labels, remove_events)

        probY = probY / np.sum(probY)  # Normalise

        # Adjust probability table based on conditional measurement data if desired

        new_state = SearchState(X, sensor_list, event_list, sensor_cost, probY)

        new_state.parent = self
        new_state.branch = k
        self.check_prop_is_valid(new_state)

        return

    def create_k_dict(self, debug: bool):
        # Define number of measurement results (k)
        #
        # Create Matrix illustrating the probability of reading a given value assuming there is a failure
        unique_k = {}
        total = 0
        # Iterate through sensors
        for j in range(len(self.feature_names)):
            if self.k_dic[j] is None:
                self.k_dic[j] = {}

            # Iterate through events
            for i in range(len(self.labels)):

                if self.X[i][j] != "":
                    # check k_array
                    # IF is in k_dic update

                    if self.k_dic[j].get(self.X[i][j]) is not None:
                        self.k_dic[j][self.X[i][j]] += self.probY[i]


                    # if it is not add extra
                    else:
                        self.k_dic[j].update({self.X[i][j]: self.probY[i]})

                        # if it is a new measurment output needed to be captured in probX add to list
                        if unique_k.get(self.X[i][j]) is None:
                            unique_k.update({self.X[i][j]: None})

    def create_condition_table(self, debug: bool):

        # create empty condition table
        self.ConditionTable = np.zeros((len(self.labels), len(self.sensor_list)))

        # Iterate through possible sensors
        # all possible sensors do adhear to the rules of the existing sensor set
        for j in range(len(self.sensor_list)):
            # Extract rule from proptable
            total = 0
            for i in range(len(self.labels)):
                p_E = self.ProbY[i][0]
                # Extract rule from proptable
                k = self.PropTable[i][j]
                total += 1

                # If not already there
                if self.k_dic[j].get(self.PropTable[i][j]) is None:
                    # Add
                    # k_dic.update({s_state.PropTable[i][j]:s_state.k_dic[j][s_state.PropTable[i][j]]})
                    self.ConditionTable[i][j] = p_E / self.k_dic[j][self.PropTable[i][j]]

                else:
                    # update
                    # k_dic[s_state.PropTable[i][j]]+=k_dic[j][s_state.PropTable[i][j]]
                    self.ConditionTable[i][j] = p_E / self.k_dic[j][self.PropTable[i][j]]

                if debug:
                    print("i = {}, j = {}, p_E/prob = {:.2f}/{:.2f} = {:.2f}".format(i, j, p_E, self.k_dic[j][
                        self.PropTable[i][j]], self.ConditionTable[i][j]))