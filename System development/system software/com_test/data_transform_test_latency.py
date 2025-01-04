import numpy as np
import time

def array_to_bytes(array):
    """Transform a NumPy array into bytes and compute the transformation time."""
    start_time = time.time()
    byte_data = array.tobytes()
    end_time = time.time()
    transformation_time = end_time - start_time
    return byte_data, transformation_time

def bytes_to_array(byte_data, shape, dtype=np.float32):
    """Transform byte data back into a NumPy array and compute the transformation time."""
    start_time = time.time()
    array = np.frombuffer(byte_data, dtype=dtype).reshape(shape)
    end_time = time.time()
    transformation_time = end_time - start_time
    return array, transformation_time

def compute_average_times(shape, dtype=np.float32, iterations=10):
    """
    Compute the average time for array-to-bytes and bytes-to-array transformations.

    Args:
        shape (tuple): Shape of the array.
        dtype: Data type of the array (default: np.float32).
        iterations (int): Number of iterations for averaging.

    Returns:
        dict: Contains total elements, total time, and time per element.
    """
    array_to_bytes_times = []
    bytes_to_array_times = []

    for _ in range(iterations):
        # Generate random array
        array = np.random.rand(*shape).astype(dtype)

        # Measure time for array-to-bytes
        byte_data, transform_to_bytes_time = array_to_bytes(array)
        array_to_bytes_times.append(transform_to_bytes_time)
    

        # Measure time for bytes-to-array
        _, transform_to_array_time = bytes_to_array(byte_data, shape, dtype)
        bytes_to_array_times.append(transform_to_array_time)

    # Total elements in the array
    total_elements = np.prod(shape)

    # Compute average times
    total_time = sum(array_to_bytes_times) + sum(bytes_to_array_times)
    time_per_element = total_time / (iterations * total_elements)

    return {
        "total_elements": total_elements,
        "total_time": total_time,
        "time_per_element": time_per_element,
    }

# Example usage
shape = (1024, 1024, 10)  # Manageable shape
dtype = np.float32        # Data type
iterations = 10           # Number of iterations

results = compute_average_times(shape, dtype, iterations)
print(f"Total elements in array: {results['total_elements']}")
print(f"Total time taken for the whole process: {results['total_time']:.6f} seconds")
print(f"Time per element: {results['time_per_element']:.12f} seconds")
