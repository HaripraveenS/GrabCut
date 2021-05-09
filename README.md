

# GrabCut — Interactive Foreground Extraction using Iterated Graph Cuts

### GrabCut 
- GrabCut is an image segmentation method based on **graph cuts**. 
- Requires a **user-specified bounding box** around the object to be segmented.  
- Estimates the color distribution of the target object and that of the  background using a **Gaussian Mixture Model**.
- Uses the GMM to construct a **Markov Random Field** over the pixel labels, with an energy function that prefers connected regions having the same label, and running a graph cut based optimization to infer their values. 
- This two-step procedure is repeated until **convergence**. 



**beta_smoothing** : 
* First find the left_difference, upleft_difference, right_difference, upright_difference arrays.
* Finds beta according to the formula mentioned in the paper.
* Find the smoothing parameter arrays based on the formulae in the the paper corresponding to the 4-neighbourhodd positions.


**construct_graph** : 
* Constructs the graph for performing mincut by building edges and graph edge capacities.
* The graph has 2 type of edges
    - **Unary Potentials:** These represent the likelihood of a pixel belonging to either the foreground or background.
    - **Pairwise Potentials:** These enforce the smoothness constraints. These edges are put between all pairs of adjacent pixels.
* Finally, uses the igraph moule to build a graph object


### Estimation Method

* Runs **st mincut** on the constructed graph to **minimise the energy**.
* Using the above GMM, a new pixel distribution is generated. That is, labels are assigned either to the foreground or the background based on the GMM and its neighbouring constraints. 
