import pandas as pd
import dowhy
from dowhy import CausalModel
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv('synthetic_causal_data.csv')

# Define SCM
causal_model = CausalModel(
    data=data,
    treatment='treatment',
    outcome='outcome',
    common_causes=['confounder']
)

# Visualize original DAG
gml_graph = causal_model._graph._graph
nx_graph = nx.DiGraph()
for edge in gml_graph.edges:
    nx_graph.add_edge(edge[0], edge[1])
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(nx_graph)
nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12)
plt.title("Causal DAG")
plt.savefig('causal_dag.png')
plt.show()

# Experiment: Try circular layout
plt.figure(figsize=(8, 6))
pos_circular = nx.circular_layout(nx_graph)
nx.draw(nx_graph, pos_circular, with_labels=True, node_color='lightgreen', node_size=500, font_size=12)
plt.title("Causal DAG (Circular Layout)")
plt.savefig('causal_dag_circular.png')
plt.show()

# Add second confounder (age)
data['age'] = np.random.normal(40, 10, len(data))
data.to_csv('synthetic_causal_data_with_age.csv', index=False)
causal_model_age = CausalModel(
    data=data,
    treatment='treatment',
    outcome='outcome',
    common_causes=['confounder', 'age']
)

# Visualize DAG with age
gml_graph_age = causal_model_age._graph._graph
nx_graph_age = nx.DiGraph()
for edge in gml_graph_age.edges:
    nx_graph_age.add_edge(edge[0], edge[1])
plt.figure(figsize=(8, 6))
pos_age = nx.spring_layout(nx_graph_age)
nx.draw(nx_graph_age, pos_age, with_labels=True, node_color='lightyellow', node_size=500, font_size=12)
plt.title("Causal DAG with Age")
plt.savefig('causal_dag_with_age.png')
plt.show()

# Experiment: Add age → treatment edge (manual addition for illustration)
nx_graph_age.add_edge('age', 'treatment')
plt.figure(figsize=(8, 6))
pos_age_manual = nx.spring_layout(nx_graph_age)
nx.draw(nx_graph_age, pos_age_manual, with_labels=True, node_color='lightcoral', node_size=500, font_size=12)
plt.title("Causal DAG with Age → Treatment")
plt.savefig('causal_dag_with_age_treatment.png')
plt.show()