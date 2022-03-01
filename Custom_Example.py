import numpy as np
import pandas as pd


def match_np_in_list(lst, comp):
    i = 0
    for arr in lst:
        if np.all(arr == comp):
            return i
        i += 1

    return -1

def get_unique(data_frame):
    """
    Error is stopped
    :return:
    """
    unique_list = []

    # print(array_np)
    arr = np.zeros(data_frame.shape[0])
    j = 0
    for i in range(data_frame.shape[0]):
        res = match_np_in_list(unique_list, data_frame.iloc[i, :])
        if res == -1:  # res is false and no match has been found
            unique_list.append(data_frame.iloc[i, :])
            arr[i] = j
            j += 1
        else:
            arr[i] = res

    new_df = pd.DataFrame(unique_list, columns=data_frame.columns)
    return (new_df, arr)

data = {
    'Sensor': [-1, 0, 1],
    'Inspection': [1, 0],
}

# create an empty dataframe
data_df = pd.DataFrame()

np.random.seed(1) #1 give unique values
# randomnly create 1000 instances
num_sensors = 3
num_inspections = 9
num_failure = 15

failure_arr = []

for i in range(num_failure):
    for j in range(num_sensors):
        data_df.loc[i, 'Sensor' + str(j)] = str(np.random.choice(data['Sensor'], 1)[0])
    for j in range(num_inspections):
        data_df.loc[i, 'Inspection' + str(j)] = str(np.random.choice(data['Inspection'], 1)[0])

    failure_arr.append("Failure "+str(i))


data_df.head()
print(data_df)

(new_df, arr) = get_unique(data_df)
print(f"Total failure coverage is {new_df.shape[0]/len(arr)*100}%")
(new_sensors_df, sensor_arr) = get_unique(data_df.filter(regex='Sensor'))
print(f"Total sensor coverage is {new_sensors_df.shape[0]/len(sensor_arr)*100}%")
(new_inspection_df, inspection_arr) = get_unique(data_df.filter(regex='Inspection'))
print(f"Total symptom coverage is {new_inspection_df.shape[0]/len(inspection_arr)*100}%")

# now we have the propagation table and the data

rel_probability = np.random.rand(num_failure)
abs_probability = rel_probability/np.sum(rel_probability)
p = 0.3 # probabitliy that there is a connection between two nodes
connection_matrix = np.empty((num_inspections,num_inspections))
connection_matrix[:] = np.NaN
random =  np.random.rand(num_inspections,num_inspections)
 # fill connection connection_matrix
for i in range(num_inspections):
    for j in range(num_inspections):
        if i!=j and random[i,j] > p:
            connection_matrix[i,j] = 10*random[i,j]



