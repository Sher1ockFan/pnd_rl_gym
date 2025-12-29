import torch, zipfile

print("torch:",torch.__version__)
print("zipfile:",zipfile.is_zipfile("policy_aug5.pt"))
torch.jit.load("policy_aug5.pt", map_location="cpu")
print("loaded successfully")