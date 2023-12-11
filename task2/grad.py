import numpy as np

def calcu_similar_grad_A(patches: np.ndarray,
                         atoms: np.ndarray, 
                         coefficients: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of 1/2||X-DA||^2 with respect to A (coefficients).
    grad_A = - D^T (X-DA)

    Args:
        patches (np.ndarray): Patches to reconstruct (X).
        atoms (np.ndarray): A bank of atoms for dictionary learning (D).
        coefficients (np.ndarray): The coefficients of the patches (A).

    Returns:
        np.ndarray: The gradient matrix with respect to A.

    Examples:
        >>> similar_grad_A = calcu_similar_grad_A(patches, atoms, coefficients)
    """
    
    grad_A = - np.matmul(atoms.T, patches - np.matmul(atoms, coefficients))

    return grad_A

def calcu_similar_grad_D(patches: np.ndarray,
                         atoms: np.ndarray, 
                         coefficients: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of 1/2||X-DA||^2 with respect to D (atoms).
    grad_D = - (X-DA) A^T

    Args:
        patches (np.ndarray): Patches to reconstruct (X).
        atoms (np.ndarray): A bank of atoms for dictionary learning (D).
        coefficients (np.ndarray): The coefficients of the patches (A).

    Returns:
        np.ndarray: The gradient matrix with respect to D.

    Examples:
        >>> similar_grad_D = calcu_similar_grad_D(patches, atoms, coefficients)
    """
    
    grad_D = - np.matmul(patches - np.matmul(atoms, coefficients), coefficients.T)

    return grad_D


def calcu_l1_grad_A(coefficients: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of ||A|| with respect to A (coefficients).
    grad_A = sign(A)

    Args:
        coefficients (np.ndarray): The coefficients of the patches (A).

    Returns:
        np.ndarray: The gradient matrix with respect to A.

    Examples:
        >>> l1_grad_A = calcu_l1_grad_A(coefficients)
    """

    grad_A = np.sign(coefficients)

    return grad_A

def calcu_huber_grad_A(coefficients: np.ndarray,
                       huber_M: float = 1.0) -> np.ndarray:
    """
    Compute the gradient of huber(A) with respect to A (coefficients).

    Args:
        coefficients (np.ndarray): The coefficients of the patches (A).
        huber_M (int): M in Huber penalty function.

    Returns:
        np.ndarray: The gradient matrix with respect to A.

    Examples:
        >>> huber_grad_A = calcu_huber_grad_A(coefficients)
    """   

    grad_A = np.where(np.abs(coefficients)<=huber_M, 2*coefficients, 2*huber_M*np.sign(coefficients))

    return grad_A


def calcu_grad_A(patches: np.ndarray,
                 atoms: np.ndarray, 
                 coefficients: np.ndarray,
                 penalty_type: str,
                 huber_M: float = 1.0,
                 penalty_weight: float = 1.0) -> np.ndarray:
    """
    Compute the gradient of the minimizing function with respect to A (coefficients).
    grad_A = similar_grad_A + penalty_weight * penalty_grad

    Args:
        patches (np.ndarray): Patches to reconstruct (X).
        atoms (np.ndarray): A bank of atoms for dictionary learning (D).
        coefficients (np.ndarray): The coefficients of the patches (A).
        penalty_type (str): The type of penalty function, can be 'l1' or 'huber'.
        huber_M (float): M in Huber penalty function.
        penalty_weight (float): The weight of penalty function.

    Returns:
        np.ndarray: The gradient matrix with respect to A.

    Examples:
        >>> grad_A = calcu_grad_A(patches, atoms, coefficients, 'huber', 1.0, 1.0)
    """
    
    similar_grad_A = calcu_similar_grad_A(patches, atoms, coefficients)
    
    if penalty_type == 'l1':
        penalty_grad_A = calcu_l1_grad_A(coefficients)
    elif penalty_type == 'huber':
        penalty_grad_A = calcu_huber_grad_A(coefficients, huber_M)

    grad_A = similar_grad_A + penalty_grad_A * penalty_weight

    return grad_A

def calcu_grad_D(patches: np.ndarray,
                 atoms: np.ndarray, 
                 coefficients: np.ndarray,
                 penalty_weight: float = 1.0) -> np.ndarray:
    """
    Compute the gradient of the minimizing function with respect to D (atoms).
    grad_D = similar_grad_D + penalty_weight * penalty_grad

    Args:
        patches (np.ndarray): Patches to reconstruct (X).
        atoms (np.ndarray): A bank of atoms for dictionary learning (D).
        coefficients (np.ndarray): The coefficients of the patches (A).
        penalty_weight (float): The weight of penalty function.

    Returns:
        np.ndarray: The gradient matrix with respect to D.

    Examples:
        >>> grad_D = calcu_grad_D(patches, atoms, coefficients, 1.0)
    """
    
    similar_grad_D = calcu_similar_grad_D(patches, atoms, coefficients)

    penalty_grad_D = 2 * atoms

    grad_D = similar_grad_D + penalty_grad_D * penalty_weight

    return grad_D

def backtracking_ls_A(descent_direction: np.ndarray,
                      patches: np.ndarray,
                      atoms: np.ndarray, 
                      coefficients: np.ndarray,
                      penalty_type: str,
                      huber_M: float = 1.0,
                      penalty_weight: float = 1.0,
                      alpha: float = 0.3,
                      beta: float = 0.7) -> float:
    """
    Backtracking line search to determine the step size of updating coefficients (A).

    Args:
        descent_direction (np.ndarray): Descent direction of the parameters.
        patches (np.ndarray): Patches to reconstruct (X).
        atoms (np.ndarray): A bank of atoms for dictionary learning (D).
        coefficients (np.ndarray): The coefficients of the patches (A).
        penalty_type (str): The type of penalty function, can be 'l1' or 'huber'.
        huber_M (float): M in Huber penalty function.
        penalty_weight (float): The weight of penalty function.
        alpha (float): Alpha in backtracking line search.
        beta (float): Beta in backtracking line search.

    Returns:
        float: Step size given by backtracking line search.

    Examples:
        >>> descent_direction = - calcu_grad_A(patches, atoms, coefficients, 'huber', 1.0, 1.0)
        >>> step_size = backtracking_ls_A(descent_direction, patches, atoms, coefficients, 'huber', 1.0, 1.0, 0.3, 0.7)
    """
    step_size = 1.0

    function_value = minimizing_function(patches, atoms, coefficients, penalty_type, huber_M, penalty_weight)
    grad_A = calcu_grad_A(patches, atoms, coefficients, penalty_type, huber_M, penalty_weight)
    
    while True:
        update_A = coefficients + step_size * descent_direction
        update_function_value = minimizing_function(patches, atoms, update_A, penalty_type, huber_M, penalty_weight)

        if update_function_value <= function_value + alpha * step_size * np.matmul(grad_A.reshape(-1,1).T, descent_direction.reshape(-1,1)):
            break

        step_size *= beta

    return step_size


def backtracking_ls_D(descent_direction: np.ndarray,
                      patches: np.ndarray,
                      atoms: np.ndarray, 
                      coefficients: np.ndarray,
                      penalty_type: str,
                      huber_M: float = 1.0,
                      penalty_weight: float = 1.0,
                      alpha: float = 0.3,
                      beta: float = 0.7) -> float:
    """
    Backtracking line search to determine the step size of updating atoms (D).

    Args:
        descent_direction (np.ndarray): Descent direction of the parameters.
        patches (np.ndarray): Patches to reconstruct (X).
        atoms (np.ndarray): A bank of atoms for dictionary learning (D).
        coefficients (np.ndarray): The coefficients of the patches (A).
        penalty_type (str): The type of penalty function, can be 'l1' or 'huber'.
        huber_M (float): M in Huber penalty function.
        penalty_weight (float): The weight of penalty function.
        alpha (float): Alpha in backtracking line search.
        beta (float): Beta in backtracking line search.

    Returns:
        float: Step size given by backtracking line search.

    Examples:
        >>> descent_direction = - calcu_similar_grad_D(patches, atoms, coefficients)
        >>> step_size = backtracking_ls_D(descent_direction, patches, atoms, coefficients, 'huber', 1.0, 1.0, 0.3, 0.7)
    """
    step_size = 1.0

    function_value = minimizing_function(patches, atoms, coefficients, penalty_type, huber_M, penalty_weight)
    grad_D = calcu_grad_D(patches, atoms, coefficients, penalty_weight/10.0)
    
    while True:
        update_D = atoms + step_size * descent_direction
        update_function_value = minimizing_function(patches, update_D, coefficients, penalty_type, huber_M, penalty_weight)

        if update_function_value <= function_value + alpha * step_size * np.matmul(grad_D.reshape(-1,1).T, descent_direction.reshape(-1,1)):
            break

        step_size *= beta

    return step_size


def minimizing_function(patches: np.ndarray,
                        atoms: np.ndarray, 
                        coefficients: np.ndarray,
                        penalty_type: str,
                        huber_M: float = 1.0,
                        penalty_weight: float = 1.0) -> float:
    """
    Compute the minimizing function value given X, D, A.

    Args:
        patches (np.ndarray): Patches to reconstruct (X).
        atoms (np.ndarray): A bank of atoms for dictionary learning (D).
        coefficients (np.ndarray): The coefficients of the patches (A).
        penalty_type (str): The type of penalty function, can be 'l1' or 'huber'.
        huber_M: M in Huber penalty function.
        penalty_weight: The weight of penalty function.

    Returns:
        float: The minimizing function value.

    Examples:
        >>> function_value = minimizing_function(patches, atoms, coefficients, 'huber', 1.0, 1.0)
    """
    function_value = np.sum((patches - np.matmul(atoms, coefficients)) * (patches - np.matmul(atoms, coefficients))) / 2.0

    # l2 penalty term for D
    function_value += np.sum(atoms * atoms) * penalty_weight/10.0
    
    # sparse penalty term for A
    if penalty_type == 'l1':
        function_value += np.sum(np.abs(coefficients)) * penalty_weight
    elif penalty_type == 'huber':
        huber = np.where(np.abs(coefficients)<=huber_M, coefficients*coefficients, 
                                                        huber_M*(np.abs(coefficients)*2 - huber_M))

        function_value += np.sum(huber) * penalty_weight

    return function_value
