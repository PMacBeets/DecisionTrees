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
        self.root = SearchState(np.array(self.inspection_df), list(self.inspection_df.columns), self.event_list, self.ave_inspection_costs, self.event_prob)


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
        self.connection_matrix[:] = np.NaN
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

    def __init__(self, PropTable, sensor_list, event_list, sensor_cost, ProbY,debug=False):
        assert isinstance(PropTable, np.ndarray)
        #assert isinstance(event_list, list)

        self.PropTable = np.array(PropTable)

        self.sensor_list = sensor_list
        self.event_list = event_list
        self.ProbY = ProbY
        self.k_dic = [{}]*self.PropTable.shape[1]  # array that hold all possible fault values for a given sensor
        self.unique_k = None
        self.SensorCost = sensor_cost # The cost of inspecting each sensor
        self.ConditionTable = None
        self.sys_entropy = None
        #self.create_condition_table(debug)

        # Sensor to be added to the sensor set on the next branch
        self.sensor = None

        self.node = None
        self.parent_node = None
        self.branch = None  # index of the branch that it branches from its parents
        self.tree_clf = None

    def lookahead(self):
        # instantiate DecisionTreeClassifier
        self.tree_clf = DecisionTreeClassifier(X=self.PropTable, feature_names=self.sensor_list, labels=self.event_list, probY=self.ProbY, sensor_cost=self.SensorCost)
        print("System entropy {:.4f}".format(self.tree_clf.entropy))
        # run algorithm id3 to build a tree
        self.tree_clf.id3(1)
        self.tree_clf.printTree()

    def buildtree(self):
        # instantiate DecisionTreeClassifier
        self.tree_clf = DecisionTreeClassifier(X=self.PropTable, feature_names=self.sensor_list, labels=self.event_list, probY=self.ProbY, sensor_cost=self.SensorCost)
        print("System entropy {:.4f}".format(self.tree_clf.entropy))
        # run algorithm id3 to build a tree
        self.tree_clf.id3()
        self.tree_clf.printTree()

    # def create_ProbX(self, debug: bool):
    #     # Define number of measurement results (k)
    #     #
    #     # Create Matrix illustrating the probability of reading a given value assuming there is a failure
    #     self.unique_k = {}
    #     total = 0
    #     # Iterate through sensors
    #     for j in range(len(self.sensor_list)):
    #
    #         # Iterate through events
    #         for i in range(len(self.event_list)):
    #
    #             if self.PropTable[i,j] != "":
    #                 # check k_array
    #                 # IF is in k_dic update
    #
    #                 if self.k_dic[j].get(self.PropTable[i,j]) is not None:
    #                     self.k_dic[j][self.PropTable[i,j]] += self.ProbY[i]
    #
    #
    #                 # if it is not add extra
    #                 else:
    #                     self.k_dic[j].update({self.PropTable[i,j]: self.ProbY[i]})
    #
    #                     # if it is a new measurment output needed to be captured in probX add to list
    #                     if self.unique_k.get(self.PropTable[i,j]) is None:
    #                         self.unique_k.update({self.PropTable[i,j]: None})
    #
    #     # create probX
    #     ProbX = np.zeros((len(self.unique_k), np.shape(self.PropTable)[1]))
    #     # Iterate through sensors
    #     for j in range(len(self.sensor_list)):
    #         # iterate through measurements
    #         for k in range(len(self.unique_k)):
    #             if self.k_dic[j].get(k) is not None:
    #                 ProbX[k][j] = self.k_dic[j][k]
    #
    #             else:
    #                 ProbX[k][j] = 0
    #     return ProbX

    # def create_condition_table(self, debug: bool):
    #
    #     # create empty condition table
    #     self.ConditionTable = np.zeros((len(self.event_list), len(self.sensor_list)))
    #
    #     # Iterate through possible sensors
    #     # all possible sensors do adhear to the rules of the existing sensor set
    #     for j in range(len(self.sensor_list)):
    #         # Extract rule from proptable
    #         total = 0
    #         for i in range(len(self.event_list)):
    #             p_E = self.ProbY[i]
    #             # Extract rule from proptable
    #             k = self.PropTable[i,j]
    #             total += 1
    #
    #             # If not already there
    #             if self.k_dic[j].get(self.PropTable[i,j]) is None:
    #                 # Add
    #                 # k_dic.update({s_state.PropTable[i][j]:s_state.k_dic[j][s_state.PropTable[i][j]]})
    #                 self.ConditionTable[i][j] = p_E / self.k_dic[j][self.PropTable[i,j]]
    #
    #             else:
    #                 # update
    #                 # k_dic[s_state.PropTable[i][j]]+=k_dic[j][s_state.PropTable[i][j]]
    #                 self.ConditionTable[i][j] = p_E / self.k_dic[j][self.PropTable[i,j]]
    #
    #             if debug:
    #                 print("i = {}, j = {}, p_E/prob = {:.2f}/{:.2f} = {:.2f}".format(i, j, p_E, self.k_dic[j][
    #                     self.PropTable[i,j]], self.ConditionTable[i][j]))
    #
    #     if debug:
    #         print("")
    #         print("Condition Table")
    #         print(self.ConditionTable)

    def splice_system_info(self, index, remove_events, k, debug: bool):
        # Remove events that do not correspond to the meaured output
        # iterate through event
        for i in range(len(self.event_list)):
            if self.PropTable[i][index] != k:
                remove_events.append(i)

        if debug:
            print("Remove Events: ", remove_events)
            print("Remove Sensor: ", index)

        # Do the following for Proptable,ProbY,ProbX,sensor_list,event_list
        # remove previous sensor
        PropTable = np.delete(self.PropTable, index, axis=1)
        sensor_list = np.delete(self.sensor_list, index)
        SensorCost = np.delete(self.SensorCost, index)

        # remove events
        PropTable = np.delete(PropTable, remove_events, axis=0)
        ProbY = np.delete(self.ProbY, remove_events, axis=0)
        event_list = np.delete(self.event_list, remove_events)

        ProbY = ProbY / np.sum(ProbY)  # Normalise

        # Adjust probability table based on conditional measurement data if desired

        new_state = SearchState(PropTable, sensor_list, event_list, SensorCost, ProbY)

        new_state.parent = self
        new_state.branch = k
        self.check_prop_is_valid(new_state)

        if debug:
            new_state.selfprint()

        return new_state

    def check_children(self,node, index, queue, debug: bool):
        # Finds all branches that result in dead ends (no known faults with this measurement)
        # Finds all branches that result in a known event
        # Return a list of known events so we can update the proptable
        # Add braches to queue whose result is not known

        remove_events = []
        # iterate through event
        for i in range(len(node.data.event_list)):
            if node.data.ConditionTable[i][index] == 1:
                node.children[node.data.PropTable[i][index]] = node.data.event_list[i]
                remove_events.append(i)
                if debug: print(
                    "For Sensor {} with measurement {} the Fault Event is {}".format(node.data.sensor_list[index],
                                                                                     node.data.PropTable[i][index],
                                                                                     node.data.event_list[i]))
        if debug:
            print("We will now add branches", end="")
            # iterate through children
            for child_key in node.children:
                if node.children[child_key] == None: print(child_key, ", ", end="")
            print("to the queue so we can further narrow down faults")

        # iterate through children
        for child_key in node.children:
            if node.children[child_key] == None:
                if debug: print("Branch = {} : Add the following state".format(child_key))
                # Get System info
                new_state = self.splice_system_info(index, remove_events.copy(), child_key, True)
                new_node = Node(new_state)
                new_node.data.parent_node = node
                node.children[child_key] = new_node
                queue.append(new_node)

    def Calculate_System_Entropy(self, s_state, log_method: str):
        if log_method not in ["2", "e"]:
            raise Exception("Log must be either base 2 or base e")

        sys_entropy = 0

        if log_method == "2":
            for y in s_state.ProbY:
                if y != 0:
                    sys_entropy += y * math.log(y, 2)
        elif log_method == "e":
            for y in s_state.ProbY:
                if y != 0:
                    sys_entropy += y * math.log(y, math.exp(1))
                    print("-{}*loge{}".format(y, y), end="")
        print("=", -sys_entropy)
        return -sys_entropy

    def Information_Gain(self, node, log_method: str, debug: bool, average: bool):
        # Calculate conditional probability and the assocaiated entropy
        # Calculate Gain form this and find best sensor to visit

        # Note: the conditionaly probabilty could be better calculated to take
        # into account previous measurements and any correlations between previously
        # observed measurements and possible future ones\

        # print("")
        # print("Node.Children1")
        # print(node.children)

        s_state = node.data
        if log_method not in ["2", "e"]:
            raise Exception("Log must be either base 2 or base e")

        num_events = np.shape(s_state.PropTable)[0]

        if debug and average:
            print("Calculating Average Information Gain with log base {}".format(log_method))
        if debug and not average:
            print("Calculating Total Information Gain with log base {}".format(log_method))

        s_state.sys_entropy = self.Calculate_System_Entropy(s_state, log_method=log_method)

        if debug: print('Total System Entropy Is {:.2f}'.format(s_state.sys_entropy))

        # Store all conditional entropy calculations
        Entropy_Arr = np.zeros(np.shape(s_state.PropTable)[1])
        Gain_Arr = np.zeros(np.shape(s_state.PropTable)[1])
        max_gain = [0, 0]  # val,index
        a = 1
        # iterate through test
        for j in range(len(s_state.sensor_list)):
            entropy = 0.0
            a_entropy = 0  # Average/Expected Entropy
            if log_method == "2":
                # iterate through event
                if debug: print("j= ", j, ": Entropy = ", end='')
                for i in range(len(s_state.event_list)):
                    if s_state.ConditionTable[i, j] != 0:
                        if average:
                            a = s_state.k_dic[j][s_state.PropTable[i][j]]
                        p = s_state.ConditionTable[i, j]
                        entropy -= a * p * math.log(p, 2)

                        if debug: print('{:.2f}*{:.2f}*log({:.2f}) + '.format(a, p, p), end='')

                Entropy_Arr[j] = entropy
                if debug: print('= {:.3}'.format(entropy))

            if log_method == "e":
                # iterate through event
                if debug: print("j= ", j, ": Entropy = ", end='')
                for i in range(len(s_state.event_list)):
                    if s_state.ConditionTable[i, j] != 0:
                        if average:
                            a = s_state.k_dic[j][s_state.PropTable[i][j]]
                        p = s_state.ConditionTable[i, j]
                        entropy -= a * p * math.log(p, math.exp(1))

                    if debug: print('{:.2f}*{:.2f}*log({:.2f}) + '.format(a, p, p), end='')

                Entropy_Arr[j] = entropy
                if debug: print('= {:.3}'.format(entropy))

        # calculate infromation gain for each
        # iterate through test
        for j in range(len(s_state.sensor_list)):
            Gain_Arr[j] = (s_state.sys_entropy - Entropy_Arr[j]) / (1.0 * s_state.SensorCost[j])
            if debug: print(
                "j= ({:.2f} - {:.2f})/{} = {:.2f}".format(s_state.sys_entropy, Entropy_Arr[j], s_state.SensorCost[j],
                                                          Gain_Arr[j]))

            if Gain_Arr[j] > max_gain[0]:
                max_gain[0] = Gain_Arr[j]
                max_gain[1] = j

        s_state.sensor = [s_state.sensor_list[max_gain[1]], max_gain[1]]  # Sensor Label & Index
        node.children = s_state.k_dic[max_gain[1]].copy()

        # print("")
        # print("Node.Children2")
        # print(node.children)

        # Create empty dictionary of pointers
        for child_key in node.children:
            node.children[child_key] = None

        if debug:
            print("")
            print("Conditional Table")
            print(s_state.ConditionTable)
            print("")
            print("Go Measure Sensor Index {}".format(max_gain[1]))
            print("Measurements of Sensor {}".format(s_state.sensor_list[max_gain[1]]))
        return max_gain[1]

    def treeprint(self, w, line_index):
        s = " "
        # Iterate through sensors
        print("")
        w_tmp = self.print_line(w, line_index)
        print(w_tmp * s, end="")
        print("    |", end="")
        for j in range(len(self.sensor_list)):
            print(f'{self.sensor_list[j]:3}', "|", end="")
        print("")

        # Iterate through events
        for i in range(len(self.event_list)):
            w_tmp = self.print_line(w, line_index)
            print(w_tmp * s, end="")
            print(f'{self.event_list[i]:3}', "|", end="")

            # Iterate through sensors
            for j in range(len(self.sensor_list)):
                print("{:3.0f} |".format(self.PropTable[i][j]), end="")
            print("")

    def print_line(self,w, line_index):
        if line_index == []:
            # print("line_index is empty")
            return w

        j = 0
        for i in range(line_index[-1] + 1):
            if i == line_index[j]:
                print("|", end="")
                j += 1
            else:
                print(" ", end="")

        return w - line_index[-1]

    def recursive_print_tree(self,node, w, line_index):

        s = " "
        w_index = 0
        print("-|--", end="")
        if type(node) is np.str_ or type(node) is str:
            print(node)

        else:
            w_index = w + 1
            w = w + 4

            node.data.treeprint(w, line_index)
            # print("line_index=",line_index)

            w = w + 5 * (len(node.data.sensor_list) + 1)

            w_tmp = self.print_line(w, line_index)
            print(w_tmp * s, "|")
            w_tmp = self.print_line(w, line_index)
            print((w_tmp) * s, node.data.sensor[0])
            w_tmp = self.print_line(w, line_index)
            print(w_tmp * s, "|")

            # print("add_index: ",line_index)
            line_index.append(w + 1)
            for k in node.children:
                w_tmp = self.print_line(w, line_index)
                print(w_tmp * s, " ", end="")
                print("{:2}".format(k), end="")
                print("-", end="")

                self.recursive_print_tree(node.children[k], w, line_index)

                # pop lines to be
            # print("remove_index: ",line_index)
            tmp = line_index.pop()
            self.print_line(w, line_index)
            print("")
            self.print_line(w, line_index)
            print("")

    def selfprint(self):

        print("      ", end="")
        # Iterate through sensors
        for j in range(len(self.sensor_list)):
            print("{}     ".format(self.sensor_list[j]), end="")
        print("")

        # Iterate through events
        for i in range(len(self.event_list)):
            print("{} |".format(self.event_list[i]), end="")

            # Iterate through sensors
            for j in range(len(self.sensor_list)):
                print("{:.3f} |".format(self.PropTable[i][j]), end="")
            print("")

        print("")
        print("SensorCost")
        print(self.SensorCost)
        print("")
        print("ProbY")
        print(self.ProbY)
        print("")
        print("K_dic")
        print(self.k_dic)
        print("")
        print("ConditionTable")
        print(self.ConditionTable)
        print("")
        # print("Children")
        # print(self.node.children)
        # print("")


    def check_prop_is_valid(self,search_state):
        if abs(np.sum(search_state.ProbY) - 1) > 0.001:
            print(search_state.ProbY)
            print("Sum is: ", np.sum(search_state.ProbY))
            raise Exception("Probability of all outcomes must sum to 1")

