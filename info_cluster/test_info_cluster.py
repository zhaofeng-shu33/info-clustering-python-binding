# pylint: disable=missing-docstring

import unittest

import networkx as nx
import numpy as np
from info_cluster import InfoCluster # pylint: disable=no-name-in-module

class TestInfoCluster(unittest.TestCase):
    def test_workflow(self):
        g = nx.Graph() # undirected graph
        g.add_edge(0, 1, weight=1)
        g.add_edge(1, 2, weight=1)
        g.add_edge(0, 2, weight=5)
        info_cluster_instance = InfoCluster(affinity='precomputed')
        info_cluster_instance.fit(g)
        info_cluster_instance.print_hierarchical_tree()
        self.assertEqual(info_cluster_instance.get_tree_depth(), 2)
    def test_4_point(self):
        g = nx.DiGraph()
        g.add_edge(0, 2, weight=2)
        g.add_edge(0, 1, weight=5)
        g.add_edge(0, 3, weight=1)
        g.add_edge(1, 2, weight=2)
        g.add_edge(1, 3, weight=1)
        info_cluster_instance = InfoCluster(affinity='precomputed')
        info_cluster_instance.fit(g)
        info_cluster_instance.print_hierarchical_tree()
        self.assertEqual(info_cluster_instance.get_tree_depth(), 3)
    def test_10_point(self):
        point_list = np.array([[0.8, 1.3], [0.9, 1.2],
                               [0.9, -0.5], [0.7, -0.6], [1.0, -0.6],
                               [-1.3, -0.5], [-1.0, -0.6], [-1.1, -0.8],
                               [-2.7, -0.8], [-2.6, -0.9]
                              ])
        info_cluster_instance = InfoCluster(gamma=1, affinity="rbf")
        info_cluster_instance.fit(point_list)
        info_cluster_instance.print_hierarchical_tree()
        # print(info_cluster_instance.critical_values)
        # print(info_cluster_instance.partition_list)
    def test_further_clustering(self):
        ic = InfoCluster(affinity='precomputed')
        g = nx.Graph()
        g.add_edge(0, 1, weight=2)
        g.add_edge(0, 2, weight=0.5)
        g.add_edge(0, 3, weight=0.5)
        g.add_edge(1, 3, weight=0.5)
        g.add_edge(1, 2, weight=0.5)
        g.add_edge(2, 3, weight=1.1)
        ic.fit(g, second_clustering=True)
        self.assertEqual(len(ic.partition_list), 4)
        self.assertAlmostEqual(ic.critical_values[0], 1)
        self.assertAlmostEqual(ic.critical_values[1], 1.1)
        self.assertAlmostEqual(ic.critical_values[2], 2)
    def test_further_clustering_data(self):
        ic = InfoCluster(gamma=1.5)
        ic.fit(np.array([[0.5, 1], [1, 0.5], [-0.1, 0], [0, -0.1]]),
               second_clustering=True)
        self.assertEqual(len(ic.partition_list), 4)
if __name__ == '__main__':
    unittest.main()
