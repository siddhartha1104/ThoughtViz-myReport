# move this code to outside of this folder and keep it inside 0Report foler 
import pickle

# Replace with your actual path
pkl_path = '../data/eeg/char/data.pkl'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Print keys
print("[INFO] Keys in data.pkl:")
print(data.keys())

# Print shapes/types of each key
for key in data:
    val = data[key]
    if hasattr(val, 'shape'):
        print(f"{key}: shape = {val.shape}, dtype = {val.dtype}")
    else:
        print(f"{key}: type = {type(val)}")
