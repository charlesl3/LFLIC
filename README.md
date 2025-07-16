# A formal relation between two disparate mathematical algorithms is ascertained from biological circuit analyses

## Abstract

We simulate and formally analyze the emergent operations from the specific anatomical layout and physiological activation patterns of a particular local excitatory-inhibitory circuit architecture that occurs throughout superficial layers of cortex.  The circuit carries out two effective procedures on its inputs, depending on the strength of its local feedback inhibitory cells.  Both procedures can be formally characterized in terms of well-studied statistical operations: clustering, and component analyses, under high-feedback-inhibition and low-feedback-inhibition conditions, respectively.  The detailed nature of these clustering and component procedures are studied in the context of extensive related literature in statistics, machine learning, and computational neuroscience.  The two operations (clustering and component analysis) have not previously been shown to contain deep connections, let alone to each be derivable from a single overarching algorithmic precursor.  The identification of this deep formal mathematical connection, which arose from the analysis of a detailed biological circuit, represents a rare instance of novel mathematical relations arising from biological analyses.



## Authors

1. **Charles(Chang) Liu**  
   - **Title:** PhD candidate  
   - **Affiliation:** Thayer School of Engineering, Dartmouth College 
   - **Contact:** charles.liu.th@dartmouth.edu

2. **Elijah FW Bowen**  
   - **Title:** Post-doctoral fellow
   - **Affiliation:** Department of Psychological and Brain Sciences, Dartmouth College
   - **Contact:** Elijah.Floyd.William.Bowen@dartmouth.edu

3. **Richard H. Granger, Jr.**  
   - **Title:** Professor
   - **Affiliation:** Department of Psychological and Brain Sciences, Dartmouth College
   - **Contact:** Richard.Granger@dartmouth.edu 




## Files and Folders

Below is a breakdown of each `.m` file and folder in the repository and their role:

1. **`ap_finding.m`**  
   *Function:* simulations to find an acute partition

2. **`Iris_Syn_visual.m`**  
   *Function:* visualizations of Iris and synthetic datasets

3. **`AIME_Iris.m`**  
   *Function:* AIME results with Iris data 

4. **`AIME_Syn.m`**  
   *Function:* AIME results with synthetic data

5. **`LIHC_Synthetic.m`**  
   *Function:* LI-HC results with the synthetic data

6. [Sample Execution](./Sample%20Execution)  
*Function:* a folder containing the executed results of the code listed above. Different file types under the same name indicate the same set of code

7. **`ConnectHypergeometric.m`**  
   *Function:* generating function for neural weight vectors that follow the hypergeometric distribution

8. **`arrow.m`**  
   *Function:* draw arrows in MATLAB plots

9. **`nmi.m`**, **`purity.m`**, **`randindex.m`**  
   *Function:* external metrics of clustering

10. **`ClusterEvalCalinskiHarabasz.m`**, **`ClusterEvalDaviesBouldin.m`**, **`ClusterEvalSilhouette.m`**  
   *Function:* internal metrics of clustering

11. **`subcluster_centroid.m`**, **`subcluster_simulate.m`**  
   *Function:* cluster generating functions to implement LI-HC

12. **`SOM.m`**  
   *Function:* an alternative clustering function other than K-means for identifying AIME components from the learned weight vectors

12. **`SetRNG.m`**  
   *Function:* random seeds settings

13. **`LIHC_MNIST.m`**  
   *Function:* LI-HC results with the MNIST data, including traditional Divisive Hierarchical Clustering results with the MNIST data (single-cycled) 

13. **`DHC_Synthetic.m`**  
   *Function:* Traditional Divisive Hierarchical Clustering results with the Synthetic data

14. **`AIME_MNIST.m`**  
   *Function:* AIME results with MNIST data (single-cycled)

## License

This project is licensed under the MIT License.
