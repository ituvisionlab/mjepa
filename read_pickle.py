import pickle

with open("b216c18a70624c308edb332ac7023860.pkl", "rb") as f:
    data = pickle.load(f)
print(data)
print(vars(data))
trainer = data.function
print(vars(trainer))  # See the Trainer object’s stored attributes
