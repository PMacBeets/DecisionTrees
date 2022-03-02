import numpy as np
import pandas as pd

path = "Data"
df = "/Proptable.csv"
inspection_df = pd.read_csv(path+df,index_col=0)
shape = inspection_df.shape
num_inspections = shape[1]
num_failure = shape[0]
event_list = list(inspection_df.index)
connection_matrix = np.array(pd.read_csv(path+"/ConnectionMatrix.csv", header=None))
assert(connection_matrix.shape[1] == connection_matrix.shape[0] == shape[1])
event_prob = np.array(pd.read_csv(path+"/ProbY.csv", header=None).transpose())
event_prob = event_prob[0,:]



from ID3_OLD import DecisionTreeClassifier

# instantiate DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(X=np.array(inspection_df), feature_names=list(inspection_df.columns), labels=event_list)
#print("System entropy {:.4f}".format(tree_clf.entropy))
# run algorithm id3 to build a tree
tree_clf.id3()
tree_clf.printTree()

