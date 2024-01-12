# IDSC (Image Denoising via Sparse Coding)

Python implementation of the Image Denoising via Sparse Coding Algorithm.

**Requirements:**
- python >= 3.6
- numpy
- pillow
- matplotlib
- h5py

## Dictionary learning model

### Initialization

We initialize the dictionary and coefficients by doing SVD of the patches. In other words, we utilize the principle components of the patches to initialize the dictionary.

$$
X = U\Sigma V^T
$$

Since computing $V$ takes a lot of time, we only compute $U$, which is done by doing eigen decomposition of $XX^T$.

$$
XX^T = U\Sigma^2 U^T
$$

Then, we select the eigen vectors of $XX^T$ with $K$ largest eigen values, denoted as $\tilde{U}$. Initialization is done by

$$
D_0 = \tilde{U} \\
A_0 = \tilde{U}^T X
$$

### Dictionary learning

$$
\textrm{minimize}_{D, A}~\frac{1}{2} \|X-DA\|_F^2 + \lambda \|A\|_1  
$$

$$
\textrm{subject~to}~ \|d_i\|_2 = 1,~ i=1,..., K
$$

Note that some tricks are applied to enhance the dictionary learning result:
- using overlapping patches
- remove the DC component (mean value of patch elements) of the patches
- make the norm of patch to be 1
- parameter continuation is used for penalty weight $\lambda$
- soft-thresholding shrinkage for updating $A$
- estimate Lipschitz constant to choose step size

## Task 1

Task 1 can be performed by running the code in "task_1_and_2":

```
cd task_1_and_2 
python main.py --data_path "../images/lena_512.png" --num_atoms 256
```

## Task 2

Task 2 can be performed by running the code in "task_1_and_2":

```
cd task_1_and_2 
python main.py --data_path "../images/McM images/McM01.tif" --num_atoms 256
```

## Task 3

In task 3, we do image denoising using learned dictionaries. That is, given learned $D$,

$$
\textrm{minimize}_{A}~\frac{1}{2} \|X-DA\|_F^2 + \lambda \|A\|_1
$$

Task 3 can be performed by running the code in "task_3/denoise":

```
cd "task_3/denoise"
python main.py --data_path "../../images/McM images/McM01_noise.mat" --clean_data_path "../../images/McM images/McM01.tif" --atoms_path "../../task_1_and_2/outputs/McM01/20240112-190718/atoms.npy"
```

### Treat each image individually
For learning dictionaries, we can treat each image individually and repeat 18 times. The results are below:
|       | Red Channel | Green Channel | Blue Channel | Average of three |
|-------|-------------|---------------|--------------|------------------|
| McM01 |    28.81    |     28.39     |    27.92     |       28.37      |       
| McM02 |    30.74    |     31.38     |    30.48     |       30.87      | 
| McM03 |    29.73    |     29.65     |    28.51     |       29.30      |
| McM04 |    30.93    |     31.84     |    29.76     |       30.85      |
| McM05 |    31.89    |     31.04     |    29.79     |       30.91      |
| McM06 |    32.28    |     31.73     |    31.30     |       31.77      |

### Put all images together
On the other hand, we can also put 18 images together and implement the denoising pipeline only one time.

To reduce the scale of problem, we employ mini-batch strategy for dictionary learning. That is, for each epoch, we randomly shuffle patches, and then learn dictionaries with small batches of patches in one optimizing step.

This dictionary learning can be performed by running the code in "task_3/learn dictionary":

```
cd "task_3/learn dictionary"
python main.py --data_path "../../images/McM images/"
```

