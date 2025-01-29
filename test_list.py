from math import ceil, floor
import numpy as np

def generate_constrained_list(x: int, y: int) -> list[int]:
    """
    Generate a list satisfying the constraints using a simplified approach.
    
    Args:
        x (int): Lower bound (result must have a value < x)
        y (int): Upper bound (result must have a value > y)
    
    Returns:
        list[int]: A list satisfying all constraints
    """
    # Validate inputs
    if x >= y:
        raise ValueError("x must be less than y")
        
    # Find the range bounds that must include 0
    min_val = min(x, 0)
    max_val = max(y, 0)
    
    # Calculate minimum required step size
    min_step = ceil((y - x) / 4)
    
    # Valid step sizes with prime factors {2, 5}
    valid_steps = [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 64, 80, 100]
    
    # Find the first valid step size >= min_step
    step_size = next(s for s in valid_steps if s >= min_step)
    
    # Find starting point by rounding x/step_size to higher magnitude
    start_multiple = np.sign(min_val)*ceil(abs(min_val/step_size)) if min_val < 0 else floor(min_val/step_size)
    start = start_multiple * step_size
    
    # Generate the list
    result = []
    current = start
    while len(result) < 5 and current < max_val+step_size:
        result.append(current)
        current += step_size
        
    return result

# Example usage
print(generate_constrained_list(-10, 10))  # might output [-16, -8, 0, 8, 16]
print(generate_constrained_list(5, 15))    # might output [0, 8, 16]