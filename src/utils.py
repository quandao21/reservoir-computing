import numpy as np


def apply_sparsity(W: np.ndarray, sparsity: float) -> np.ndarray:
    """
    Applies sparsity to the given matrix W.

    Parameters:
        - W: The matrix to which sparsity will be applied.
        - sparsity: The fraction of elements to be zeros (0 <= sparsity < 1).

    Returns:
        The modified matrix with applied sparsity
    """
    total_elements = W.size
    num_nonzero = int(total_elements * sparsity)
    indices = np.random.choice(
        np.arange(total_elements), size=num_nonzero, replace=False
    )
    W.flat[indices] = 0
    return W


def has_valid_eigenvalues(W: np.ndarray) -> bool:
    """
    Checks if the matrix W has no zero eigenvalues.

    Parameters:
        - W: The input matrix whose eigenvalues will be checked.

    Returns:
        True if the matrix has non-zero eigenvalues, otherwise False.
    """
    eigenvalues = np.linalg.eigvals(W)
    return max(abs(eigenvalues)) > 0


def scale_matrix(W, spectral_radius):
    """
    Scales the matrix W to achieve the desired spectral radius.

    Parameters:
        - W: The matrix to scale.
        - spectral_radius: The desired spectral radius.

    Returns:
        The scaled matrix, or None if scaling is not possible due to zero eigenvalues.
    """
    eigenvalues = np.linalg.eigvals(W)
    max_eigenvalue = max(abs(eigenvalues))

    if max_eigenvalue == 0:
        return None  # Return None if the max eigenvalue is zero (invalid matrix)

    return W * (spectral_radius / max_eigenvalue)
