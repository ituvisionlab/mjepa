import pickle

with open("/gpfs/home/unalg01/jepa/job_57718343/57718343_2_result.pkl", "rb") as f:
    data = pickle.load(f)
print(data)
print(vars(data))
trainer = data.function
print(vars(trainer))  # See the Trainer object’s stored attributes
