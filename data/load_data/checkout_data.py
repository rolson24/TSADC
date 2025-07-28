import pickle
import sys

def load_array_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        array = pickle.load(file)
    return array

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python checkout_data.py <path_to_pickle_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    data = load_array_from_pickle(file_path)

    data_trace = data.get_data()

    print(data_trace)  # Print the loaded data
    
    try:
        print("Array shape:", data_trace.shape)
        print("I think the first dimension is the number of sensors, and the second dimension is the time points.")
    except AttributeError:
        print("The loaded object does not have a 'shape' attribute.")