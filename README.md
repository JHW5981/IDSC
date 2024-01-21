# IDSC (Image Denoising via Sparse Coding)

Python implementation of the Image Denoising via Sparse Coding Algorithm. 

<span style="color:red; font-weight:bold;font-size:24px;"> Due to the large size of the experimental results file, please visit https://github.com/sjtu-jhw/IDSC to view all our results.  </span>



**Requirements:**
- python >= 3.6
- numpy
- pillow
- matplotlib
- h5py

## Code Structure
<pre>
├── images/  # images that we need to reconstruct and denoise
├── task_1_and_2/  # experimental results for task1-2
│   └── outputs/
│       ├── McM01/20240112-203951 # best checkpoint for McM01
│       │   ├── atoms.npy # atoms matrix we learned, i.e. D
│       │   ├── atoms.png # visualization of atoms 
│       │   ├── coefficients.npy # coefficients matrix we learned, i.e. A
│       │   ├── reconstruct.npy # reconstructed image, i.e. DA
│       │   ├── reconstruct.png # visualization of reconstructed image 
│       │   └── train.log # training process logs
│       ├── ...
│       └── McM18/20240113-095254 # best checkpoint for McM18
│   ├── main.py
│   └── utils.py
│
├── task_3/  # experimental results for task3
│   ├── learn dictionary/ # find dictionary from clean images
│   └── denoise/ # denoising results for task3
│       ├── outputs/...
│       ├── main.py
│       └── utils.py
│
├── task_4/  # experimental results for task4
│   ├──outputs/...
│   ├── hyper_search.py # search for optimal hyper-parameters
│   ├── main.py
│   └── utils.py
│
├── visualization/ # denoising visualization results
├── .gitattributes
├── LICENSE
└── README.md
</pre>


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
\textrm{min}_{D, A}~\frac{1}{2} \|X-DA\|_F^2 + \lambda \|A\|_1  
$$

$$
\textrm{s.t.} ~~ \|d_i\|_2 = 1,~ i=1,..., K
$$

Note that some tricks are applied to enhance the dictionary learning result:
- use overlapping patches
- remove the DC component (mean value of patch elements) of the patches
- make the norm of patch to be 1
- parameter continuation is used for penalty weight $\lambda$
- soft-thresholding shrinkage for updating $A$
- estimate Lipschitz constant to choose step size


## Task 1

For optimal performance, we identified the best hyperparameters in tasks 1-3:
- penalty_weight &nbsp;&nbsp;&nbsp;&nbsp; $70$
- max_iters  &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $2000$
- patch_size &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $8\times8$
- patch_gap &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $2$
- num_atoms &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $128$

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
\textrm{min}_{A}~\frac{1}{2} \|X-DA\|_F^2 + \lambda \|A\|_1
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
| McM01 |    28.93    |     28.37     |    28.20     |       28.50      |       
| McM02 |    30.85    |     31.47     |    30.46     |       30.93      | 
| McM03 |    30.01    |     29.92     |    28.96     |       29.63      |
| McM04 |    31.85    |     32.20     |    30.59     |       31.55      |
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


## Task 4

In task 4, the dictionary is learned from the noisy images. We perform dictionary learning and image denoising in the same time. This can be performed by running the code in "task_4":

```
cd task_4
python main.py --data_path "../images/McM images/McM01_noise.mat" --clean_data_path "../images/McM images/McM01.tif" --num_atoms 169
```

We conduct hyper-parameter search to find best hyper-parameters so that the highest psnr can be achieved. The reference code locates at `./task_4/hyper_search.py`. Finally, for task4 our best hyper-parameters are:
- penalty_weight &nbsp;&nbsp;&nbsp;&nbsp; $20$
- max_iters  &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $500/1000$ 
- patch_size &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $16\times16$
- patch_gap &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $1$
- num_atoms &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $256$

The best results are below:
|       | Red Channel | Green Channel | Blue Channel | Average of three |
|-------|-------------|---------------|--------------|------------------|
| McM01 |    28.14    |     27.58     |    27.21     |       27.64      |       
| McM02 |    30.36    |     30.91     |    29.98     |       30.41      | 
| McM03 |    29.23    |     29.13     |    27.99     |       28.79      |
| McM04 |    30.96    |     31.74     |    29.79     |       30.83      |
| McM05 |    31.23    |     30.33     |    29.08     |       30.21      |
| McM06 |    31.68    |     31.21     |    30.73     |       31.21      |
| McM07 |    31.05    |     31.11     |    30.51     |       30.89      |
| McM08 |    31.70    |     32.17     |    31.86     |       31.91      |
| McM09 |    30.56    |     31.47     |    31.18     |       31.07      |
| McM10 |    31.67    |     31.81     |    31.50     |       31.66      |
| McM11 |    32.09    |     31.61     |    32.63     |       32.11      |
| McM12 |    32.72    |     31.65     |    31.71     |       32.03      |
| McM13 |    33.91    |     34.23     |    33.11     |       33.75      |
| McM14 |    32.81    |     33.72     |    31.95     |       32.83      |
| McM15 |    31.93    |     33.00     |    32.93     |       32.62      |
| McM16 |    29.46    |     28.09     |    30.55     |       29.36      |
| McM17 |    29.17    |     29.14     |    29.50     |       29.27      |
| McM18 |    29.72    |     29.56     |    31.53     |       30.27      |
