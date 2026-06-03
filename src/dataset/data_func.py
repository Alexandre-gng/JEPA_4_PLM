import random


def mask_sequence(sequence, mask_ratio=0.15):
    """
    Mask a percentage of amino acids in the input sequence with 'X'.
    MASKED_AMINO_ACIDS = ['X']  
    
    Args:
        sequence (str): The input protein sequence.
        mask_ratio (float): The percentage of amino acids to mask (default is 0.15 for 15%).
    Returns:
        str: The masked protein sequence.

    """
    sequence = list(sequence)
    num_to_mask = int(len(sequence) * mask_ratio)
    mask_indices = random.sample(range(len(sequence)), num_to_mask)
    for idx in mask_indices:
        sequence[idx] = "X"
    return "".join(sequence)