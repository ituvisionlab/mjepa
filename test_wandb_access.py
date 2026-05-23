# test_wandb_access.py
import requests

try:
    print("Trying to access https://api.wandb.ai ...")
    r = requests.get("https://api.wandb.ai", timeout=10)
    print("Success. Status code:", r.status_code)
except Exception as e:
    print("Failed to access WandB:", e)
