# Local Mahalanobis distance

Mahalanobis distance is a measure of the distance between two points in a multivariate space.

$$d_M(X,Y)=\sqrt{(X-Y)^T\Sigma^{-1} (X-Y)}$$

Here, we propose a new index named local Mahalanobis distance (LMD) to describe the distance between a point $\mathrm{x}$ and a region represented as a series of points.

$$D(\mathrm{x};S)=\min_k\{d_M(\mathrm{x},C_k)|C_k\in S\}$$

Therefore, we first generate a series of anchors from the training samples as a representation of the target region and then calculate the LMD value for a new sample point. 

Based on the LMD index, we further developed different solutions for three fault diagnosis tasks: fault detection, fault root cause isolation, and fault severity assessment.



## Usage

```python
from algorithm.LMD import LMD

ref = np.random.randn(1000, 5)
sig = np.random.randn(1000, 5)+5
lmdM = LMD(alpha=0.99,minR='auto',Ulimit=2,EPDweight = 0.5)
lmdM.fit(ref)   
Dl = lmdM.transform(sig) 
```

## Example 

Run main/demo.py



## Reference

- [J. Yang and C. Delpha. An Incipient Fault Diagnosis Methodology Using Local Mahalanobis Distance: Detection Process Based on Empirical Probability Density Estimation. Signal Processing, 2022, vol. 190, p. 108308. ](https://doi.org/10.1016/j.sigpro.2021.108308)

- [J. Yang and C. Delpha. An Incipient Fault Diagnosis Methodology Using Local Mahalanobis Distance: Fault Isolation and Fault Severity Estimation. Signal Processing, 2022, p. 108657.](https://doi.org/10.1016/j.sigpro.2022.108657)

