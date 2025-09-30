import pickle

with open("pickle_files/notebook.pkl", "rb") as f:
    data = pickle.load(f)

print("Type:", type(data))
try:
    print("Keys:", data.keys())
except:
    print("Preview:", data)
