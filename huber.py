import numpy as np
from typing import List, Tuple

class Huber:
    def __init__(self, A, D, Y, 
                            delta, 
                            alpha=1.0,
                            max_iter=5000):
        self.A = A
        self.D = D
        self.Y = Y
        self.delta = delta
        self.alpha = alpha
        self.max_iter = max_iter
    def _huber_loss_and_gradient(self, A, D, Y, delta, 
                                alpha=1.0,
                                reweighted_alpha=None) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Returns the Huber loss and the gradients with respect to A and D.

        Args:
            A (np.ndarray): 
                Coefficient matrix with shape (`D.shape[1]`, `Y.shape[1]`, `n_channels`(optional)).
                `D.shape[1]` is the number of atoms.
                `Y.shape[1]` is the number of patches.
                `n_channels` is the channel of image, default=3(optional)

            D (np.ndarray): 
                Dictionary matrix with shape (`D.shape[0]`, `D.shape[1]`, `n_channels`(optional))
                `D.shape[0]` is the dimension of each atom, e.g. if patch shape is (8, 8), the dimension is 8x8=64.
                `D.shape[1]` is the number of atoms, e.g. `K`=128.
                `n_channels` is the channel of image, default=3(optional)

            Y (np.ndarray): 
                Target matrix with shape (`Y.shape[0]`, `Y.shape[1]`, `n_channels`(optional))
                `Y.shape[0]` is the dimension of each atom, e.g. if patch shape is (8, 8), the dimension is 8x8=64. It is equal to `D.shape[0]`.
                `Y.shape[1]` is the number of patches, e.g. if image size is (512, 512), patch size is (8, 8), the number of patches is (512/8)**2=4096.
                `n_channels` is the channel of image, default=3(optional)

            delta (float): 
                Robustness of the Huber estimator.

            alpha (float): 
                Regularization parameter of the Huber estimator.   

            reweighted_alpha (np.ndarray): 
                Reweighted alpha array with shape (`A.shape[1]`, ).
                Each element `alpha_i` represents the weighted weight for the corresponding column `i` in matrix `A`.
                Defaults to `np.ones(A.shape[1])` if not provided.
                
        Returns:
            loss (float):
                Huber loss.

            grad_wrt_D (np.ndarray): 
                D_gradient matrix with shape like `D.shape`
                Returns the derivative of the Huber loss with respect to `D`.

            grad_wrt_A (np.ndarray): 
                A_gradient matrix with shape like `A.shape`
                Returns the derivative of the Huber loss with respect to `A`.
        """

        # Caculate main square loss
        linear_loss = Y - np.einsum("ijk,jmk->imk", D, A) if len(Y.shape)==3 else Y - np.einsum("ij,jm->im", D, A)
        main_squared = np.sum(linear_loss**2, axis=(0,1))

        # Caculate huber loss of A
        abs_A = np.abs(A)
        outliers_mask = abs_A > delta
        # Calculate the linear loss due to the outliers. Square loss for none-outliers 
        huber_loss_A = np.where(outliers_mask, delta * (abs_A - delta / 2), A**2 / 2)
        huber_loss_A_sum = np.sum(huber_loss_A, axis=(0, 1))

        # Combine losses
        loss = (main_squared/2 + huber_loss_A_sum)/(huber_loss_A.shape[0]*huber_loss_A.shape[1])

        # Caculate gradient
        grad_wrt_D = np.zeros_like(D)
        grad_wrt_A = np.zeros_like(A)

        # Gradient with respect to D
        grad_wrt_D += -np.einsum("ijk,jmk->imk", linear_loss, A.transpose((1,0,2)))/(huber_loss_A.shape[0]*huber_loss_A.shape[1]) if len(Y.shape)==3 \
                    else -np.einsum("ij,jm->im", linear_loss, A.transpose((1,0)))/(huber_loss_A.shape[0]*huber_loss_A.shape[1])

        # Gradient with respect to A
        # Main part
        grad_wrt_A += -np.einsum("ijk,jmk->imk", D.transpose((1,0,2)), linear_loss)/(huber_loss_A.shape[0]*huber_loss_A.shape[1]) if len(Y.shape)==3 \
                    else -np.einsum("ij,jm->im", D.transpose((1,0)), linear_loss)/(huber_loss_A.shape[0]*huber_loss_A.shape[1])
        # Regularization part
        signed_arr = np.where((outliers_mask) & (A < 0), -1, 1)
        grad_wrt_A += alpha * np.where(outliers_mask, delta*signed_arr, A)/(huber_loss_A.shape[0]*huber_loss_A.shape[1])

        return (loss, grad_wrt_D, grad_wrt_A)

    def train(self):
        self.loss = []
        for i in range(self.max_iter):
            loss, grad_wrt_D, grad_wrt_A = self._huber_loss_and_gradient(self.A, self.D, self.Y, self.delta, self.alpha)
            self.loss.append(loss)
            print(f"iter: {i}/{self.max_iter}, loss: {loss}")
            # if i >= 1 and abs(self.loss[-2] - self.loss[-1]) <= 1e-6:
            #     print("no valid gradient decent occurs, break.")
            #     break
            if len(self.Y.shape)==2:
                self.A -= 150*grad_wrt_A
                self.D -= 150*grad_wrt_D
            else:
                self.A -= 100*grad_wrt_A
                self.D -= 100*grad_wrt_D           
