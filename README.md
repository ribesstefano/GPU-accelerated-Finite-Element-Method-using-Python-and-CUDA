<a target="_blank" href="https://colab.research.google.com/github/ribesstefano/GPU-accelerated-Finite-Element-Method-using-Python-and-CUDA/blob/8e681149dd77ca31f6bf4182264a5f185e3beae9/TRA105_GPU_accelerated_Computational_Methods_using_Python_and_CUDA.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# GPU Computing with Python - FEM

This repository includes the work done within the course _TRA105 - GPU-accelerated Computational Methods using Python and CUDA_, held at Chalmers University.
The main contributions are given by [Stefano Ribes](mailto:ribes.stefano@gmail.com), who developed all the high performance code, [Kim Louisa Auth](mailto:kim.auth@chalmers.se), who wrote an initial version of the FEM algorithm, and [Fredrik Larsson](mailto:Fredrik.Larsson@chalmers.se), who supervised the project.

The Jupyter Notebook [TRA105_GPU_accelerated_Computational_Methods_using_Python_and_CUDA.ipynb](TRA105_GPU_accelerated_Computational_Methods_using_Python_and_CUDA.ipynb) includes most of the project work and can be run in Google Colab.

In the first part of the notebook, we describe the FEM algorithm from a higher point of view. We then report a simple mechanism to generate "large enough" FEM problems. Then, we proceed to evaluate different solver strategies and select the best performing solver algorithm.
At this point, we present three different implementation of the K-assembly step in the FEM algorithm. An additional K-assembly implementation based on cell coloring is proposed in the Appendix as a work in progress.
In the end, we show an evaluation of the three different proposed K-assembly steps, before drawing our conclusions.

## Methodology

The FEM algorithm is mainly divided in two phases: the assembly of stiffness matrix $K$ and the linear solver part.

### K-Assembly

In this work, we proposed four different implementations for computing the assembly of the stiffness matrix:

* Naïve CPU implementation
* Batched CPU implementation via Numpy and Numba
* Batched GPU implementation via CuPy
* Custom CUDA kernel implementation via Numba

### Solver Profiling

As we can see in the following figure, the `minres` solver is the best performing one, especially in the case of large matrix dimensions. We believe that the main reason for such result lies in the fact that `minres` is able to best leverage the symmetry and the sparseness of the stiffness matrix compared to the other solver algorithms.

![image](https://user-images.githubusercontent.com/17163014/222154915-b5b35c26-875c-49fd-94e9-c67a8ac8d744.png)

## Evaluation and Results

The tests have been conducted over precomputed grids of up to 5 million nodes. The measurements were collected on different devices, namely an Intel 8-cores Xeon processor, an Nvidia Tesla T4 and an Nvidia RTX 2080 Ti.

As expected, the GPU implementation is clearly superior in terms of performance, being as low as 3% of the CPU time and $8.2\times$ faster than CPU on average, with a peak of $27.2\times$.

![normalized_execution_time](https://user-images.githubusercontent.com/17163014/222154686-e93d7706-0151-44b2-b588-cc86632de05b.png)

## Final Remarks

The aforementioned notebook contains a more in-depth description and analysis of the proposed designs. It also includes additional notes and future work proposals.

