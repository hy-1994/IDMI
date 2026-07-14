# IDMI: An effective method for metacells inference from single-cell transcriptome data

### Overview：

We propose an effective metacells inference method named IDMI. It adopts the maximum minimum sampling method to select waypoints, utilizes the increment of diversity as the measure of the similarity between metacells, and realizes metacells inference with the minimum diversity increment algorithm. The IDMI, as a method for handling missing values in scRNA-seq data, helps improve the integrity and accuracy of the data and provides a reliable basis for subsequent data analysis.

### Systems Requirements

The scripts were written in Python language.

To run the scripts, you need several python packages, as follows:

- sys

- numpy

- pandas

- scanpy

- numba

- tqdm

- math

- seaborn

- matplotlib

  ​

Recommended install command:

conda create -n idti -c conda-forge -c bioconda python=3.7 scanpy matplotlib scikit-learn pygraphviz
