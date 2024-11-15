import pickle

with open("/gpfs/home/unalg01/jepa/job_54290104/54290104_submitted.pkl", "rb") as f:
    data = pickle.load(f)
print(data)
print(vars(data))
trainer = data.function
print(vars(trainer))  # See the Trainer object’s stored attributes
