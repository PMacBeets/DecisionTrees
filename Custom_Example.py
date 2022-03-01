import numpy as np
import pandas as pd
import ID3_Plus
import networkx as nx
import graphviz


system = ID3_Plus.InspectionSystem()
system.root.buildtree()
h = graphviz.Digraph('H', filename='InspectionGraph.gv')

#illustrate graph
node = system.root.tree_clf.node
def build_node_tree(node):
    # if node.childs is [] and node.next is None:
    #     return
    if node.childs is not None:
        for child in node.childs:
            print(node.name, child.name)
            h.edge(node.name, child.name)
            build_node_tree(child)

    elif node.next is not None:
        if node.next.name is None:
            h.edge(node.name, node.next.value)
        else:
            h.edge(node.name, node.next.name)
            build_node_tree(node.next)
    else:
        return

build_node_tree(node)
h.view()