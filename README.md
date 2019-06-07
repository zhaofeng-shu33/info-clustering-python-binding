[![Travis](https://api.travis-ci.com/zhaofeng-shu33/info-clustering-python-binding.svg?branch=master)](https://travis-ci.com/zhaofeng-shu33/info-clustering-python-binding)
[![Appveyor](https://ci.appveyor.com/api/projects/status/github/zhaofeng-shu33/info-clustering-python-binding?branch=master&svg=true)](https://ci.appveyor.com/project/zhaofeng-shu33/info-clustering-python-binding)
# Python binding
[![PyPI](https://img.shields.io/pypi/v/info_cluster.svg)](https://pypi.org/project/info_cluster)
Disabled by default. The binding requires boost-python library. To enable it, run `cmake` with `-DUSE_PYTHON=ON`
To make it independent of boost dynamic library, static linking should be enabled in CMAKE configuration.
To package the library, use `python setup.py bdist_wheel`.
Install the package by `pip install --user info_cluster`. If your system cmake is called cmake3, you can use
CMAKE=cmake3 pip install --user info_cluster`.
Below is the prebuild
binary packages:

| Platform | py3.6 | py3.7 |
| -------- | :---: | :---: |
| Windows  |   T   |       |
| MacOS    |       |       |
| Linux    |       |       |

## Demo code
![](example.png)
We provide a high-level wrapper of info-clustering algorithm. 
After installing `info_cluster`, you can use it as follows:
```Python
from info_cluster import InfoCluster
import networkx as nx
g = nx.Graph() # undirected graph
g.add_edge(0, 1, weight=1)
g.add_edge(1, 2, weight=1)
g.add_edge(0, 2, weight=5)
ic = InfoCluster(affinity='precomputed') # use precomputed graph structure
ic.fit(g)
ic.print_hierachical_tree()
```
The output is like
```shell
      /-0
   /-|
--|   \-2
  |
   \-1
```
```Python
import psp # classify the three data points shown in the above figure
g = psp.PyGraph(3, [(0,1,1),(1,2,1),(0,2,5)]) # index started from zero, similarity is 5 for vertex 0 and 2
g.run() # use maximal flow algorithm to classify them
print(g.get_critical_values()) # [2,5]
print(g.get_category(2)) # get the result which has at least 2 categories, which is [1,0,1]
```    

## Parametric Dilworth Truncation(pdt) implementation
We provide another alternative implementation, which can be used similar to **PyGraph**.
You shold apply a patch [preflow.patch](./preflow.patch) to `preflow.h`, which belongs to lemon library 1.3.1, see
[#625](https://lemon.cs.elte.hu/trac/lemon/ticket/625).

```Python
import psp
g = psp.PyGraphPDT(3, [(0,1,1),(1,2,1),(0,2,5)]) # index started from zero, similarity is 5 for vertex 0 and 2
g.run() # use maximal flow algorithm to classify them
print(g.get_critical_values()) # [2,5]
print(g.get_category(2)) # get the result which has at least 2 categories, which is [0,1,0]
```  