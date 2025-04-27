import hashlib
from comp_capacity.repr.network import MatrixContainer


def generate_hash(matrices: MatrixContainer) -> str:
    """
    Generate a hash for the given matrices. Provides a unique
    string for each matrix configuration. Allows for computationally cheap
    deduplication of networks with the same topology.    

    Args:
        matrices (MatrixContainer): The matrices to hash.
    Returns:
        str: The hexadecimal digest of the hash.
    """
    # Convert the matrices to a string representation
    matrix, _ = matrices.concat()
    
    _bytes = matrix.cpu().numpy().tobytes()
    # Create a hash object
    hashed = hashlib.sha256(_bytes).hexdigest()
    
    # Return the hexadecimal digest of the hash
    return hashed