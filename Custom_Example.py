import numpy as np
import pandas as pd
import ID3_Plus
import networkx as nx
import graphviz
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import os

path = "Data"
system = ID3_Plus.InspectionSystem(path)
system.buildtree(True)
h = graphviz.Digraph('H', filename='InspectionGraph.gv')
#h = graphviz.AGraph("H", filename='InspectionGraph.gv', directed=True)

#illustrate graph
node = system.tree_clf.node
def build_node_graph(node, h):
    # if node.childs is [] and node.next is None:
    #     return
    node.create_df()
    if node.childs is not None:
        for child in node.childs:
            #print(node.name, child.name)
            h.node(node.uid, label=node.name)
            h.edge(node.uid, child.uid)
            build_node_graph(child, h)

    elif node.next is not None:
        if node.next.name is None:
            h.node(node.next.uid, label=node.name)
            h.edge(node.uid, node.next.uid)
            #node.create_df()
        else:
            h.node(node.next.uid, label=node.name)
            h.edge(node.uid, node.next.uid)
            build_node_graph(node.next, h)
    else:
        return

def build_node_tree2(node, h):
    # if node.childs is [] and node.next is None:
    #     return
    node.create_df()
    #print(f"node.uid={node.uid}")
    if node.childs is not None:
        #print(f"node.childs ={[n.uid for n in node.childs]}")
        for child in node.childs:
            #print(node.name, child.name)
            h.node(child.uid, label=str(child.value))
            h.edge(node.uid,child.uid)
            #h[child.uid] = Node(child.value, parent=h[node.uid])
            build_node_tree2(child, h)
    elif node.next is not None:
        #print(f"node.next={node.next.uid}")
        if node.next.name is None:
            h.node(node.next.uid, label=node.next.value)
            h.edge(node.uid,node.next.uid,label=f"p={node.ss.probY[node.ss.labels.index(node.next.value)]:.2f}") #label=f"p={node.ss.probY[node.ss.labels.index(node.next.value)])
            #h[node.uid] = Node(node.next.value, parent=h[node.uid])
            #node.create_df()
        else:
            h.node(node.next.uid, label=node.next.value)
            h.edge(node.uid,node.next.uid,label=str(node.next.cost))
            #h[node.next.uid] = Node(node.next.value, parent=h[node.uid])
            build_node_tree2(node.next, h)
    else:
        return

#def create_tree_graph(constraint):
def build_node_tree(node, nodes):
    # if node.childs is [] and node.next is None:
    #     return
    node.create_df()
    #print(f"node.uid={node.uid}")
    if node.childs is not None:
        #print(f"node.childs ={[n.uid for n in node.childs]}")
        for child in node.childs:
            #print(node.name, child.name)
            nodes[child.uid] = Node(child.value, parent=nodes[node.uid])
            build_node_tree(child, nodes)
    elif node.next is not None:
        #print(f"node.next={node.next.uid}")
        if node.next.name is None:
            nodes[node.uid] = Node(node.next.value, parent=nodes[node.uid])
            #node.create_df()
        else:
            nodes[node.next.uid] = Node(node.next.value, parent=nodes[node.uid])
            build_node_tree(node.next, nodes)
    else:
        return
h.node(str(-1), label="Start")
h.node(node.uid, label=node.value)
h.edge(str(-1),node.uid, label=str(node.cost))
build_node_tree2(node, h)
nodes = {}
nodes[node.uid] = Node(node.name)
build_node_tree(node, nodes)
for pre, fill, nod in RenderTree(nodes[node.uid]):
    print("%s%s" % (pre, nod.name))

# A.layout()  # layout with default (neato)
# A.draw('simple.png',prog='dot') # draw png


#DotExporter(nodes[node.uid]).to_dotfile("InspectionTree.dot")
#graphviz.render('dot', 'png', 'InspectionTree.dot')
#diagraph_tree.render(directory='doctest-output', view=True)
h.view()
