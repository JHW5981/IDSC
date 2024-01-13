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
D_0 = \tilde{U} 
$$

$$
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
| McM01 |    28.81    |     28.41     |    27.92     |       28.38      |       
| McM02 |    30.75    |     31.40     |    30.53     |       30.89      | 
| McM03 |    29.77    |     29.69     |    28.55     |       29.34      |
| McM04 |    30.93    |     31.85     |    29.76     |       30.85      |
| McM05 |    31.90    |     31.05     |    29.81     |       30.92      |
| McM06 |    32.35    |     31.78     |    31.38     |       31.84      |
| McM07 |    31.38    |     31.45     |    30.88     |       31.23      |
| McM08 |    32.50    |     33.16     |    32.78     |       32.81      |
| McM09 |    31.38    |     32.34     |    32.16     |       31.96      |
| McM10 |    32.45    |     32.63     |    32.37     |       32.48      |
| McM11 |    33.03    |     32.50     |    34.25     |       33.26      |
| McM12 |    34.33    |     33.42     |    33.48     |       33.75      |
| McM13 |    36.30    |     36.96     |    35.00     |       36.09      |
| McM14 |    34.49    |     35.31     |    33.69     |       34.50      |
| McM15 |    33.02    |     34.55     |    34.43     |       34.00      |
| McM16 |    30.06    |     28.60     |    31.11     |       29.93      |
| McM17 |    29.82    |     29.73     |    30.03     |       29.86      |
| McM18 |    29.68    |     29.57     |    31.39     |       30.21      |

### Put all images together
On the other hand, we can also put 18 images together and implement the denoising pipeline only one time.

To reduce the scale of problem, we employ mini-batch strategy for dictionary learning. That is, for each epoch, we randomly shuffle patches, and then learn dictionaries with small batches of patches in one optimizing step.

This dictionary learning can be performed by running the code in "task_3/learn dictionary":

```
cd "task_3/learn dictionary"
python main.py --data_path "../../images/McM images/"
```

The results are below:
|       | Red Channel | Green Channel | Blue Channel | Average of three |
|-------|-------------|---------------|--------------|------------------|
| McM01 |    27.52    |     27.01     |    26.66     |       27.06      |       
| McM02 |    29.40    |     29.77     |    29.06     |       29.41      | 
| McM03 |    28.30    |     28.33     |    27.01     |       27.88      |
| McM04 |    29.35    |     30.64     |    27.96     |       29.31      |
| McM05 |    30.24    |     29.30     |    28.03     |       29.19      |
| McM06 |    30.63    |     29.95     |    29.62     |       30.07      |
| McM07 |    29.85    |     29.73     |    29.10     |       29.56      |
| McM08 |    30.43    |     31.01     |    30.71     |       30.72      |
| McM09 |    29.32    |     30.51     |    30.14     |       29.99      |
| McM10 |    30.60    |     30.72     |    30.54     |       30.62      |
| McM11 |    30.96    |     30.61     |    31.62     |       31.06      |
| McM12 |    31.76    |     30.27     |    30.56     |       30.86      |
| McM13 |    33.00    |     33.37     |    32.70     |       33.02      |
| McM14 |    31.86    |     32.67     |    31.11     |       31.88      |
| McM15 |    30.76    |     32.15     |    32.09     |       31.67      |
| McM16 |    28.66    |     27.25     |    28.91     |       28.27      |
| McM17 |    28.25    |     28.21     |    28.35     |       28.27      |
| McM18 |    28.81    |     28.67     |    29.29     |       28.92      |