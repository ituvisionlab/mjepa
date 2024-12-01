import pickle

with open("/gpfs/home/unalg01/jepa/job_55314001/55314001_0_result.pkl", "rb") as f:
    data = pickle.load(f)
print(data)
print(vars(data))
trainer = data.function
print(vars(trainer))  # See the Trainer object’s stored attributes
