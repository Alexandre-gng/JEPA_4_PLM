import torch
from torch.utils.data import DataLoader

from JEPA.JEPA import JEPA
from JEPA.train_JEPA import test_jepa, test_1sequence

from dataset.ProteinDataset import ProteinDataset
from main import collate_sequences

TEST_PATH = 'dataset/test'


# === Model loading ===
print("Loading the model and running the test function on a single sequence...")
# load the .pt file and run the test function on a single sequence
jepa = JEPA(latent_dim=320, output_dim=320, tau=0.99)
print("Loading the model from 'models/10k_SIGReg/jepa_model.pt'...")
jepa.load_state_dict(torch.load('models/10k_SIGReg/jepa_model.pt', map_location=torch.device('cpu')))
print("Model loaded successfully.")


# === Test set loading ===
test_dataset = ProteinDataset(root_path=TEST_PATH, masked_ratio=0.15, n_sequences=200)
print(f"Test dataset loaded with {len(test_dataset)} sequences.")
# Savec the test dataset in a .pt file for later use
torch.save(test_dataset, 'test_dataset.pt')
print("Test dataset saved to 'test_dataset.pt'.")

"""
test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_sequences,
        num_workers=0,
        drop_last=False,
    )
"""



# === Run test function and save latents ===
l_results = []
for i in range(len(test_dataset)):
    print(f"prot {i+1}/{len(test_dataset)}:")
    full_seq, masked_seq, _ = test_dataset[i]
    context_latent, target_latent = test_1sequence(jepa, full_seq, masked_seq, device=torch.device('cpu'))
    print(f"Context latent shape: {context_latent.shape}, Target latent shape: {target_latent.shape}")
    # Save the context and target latents in a list of dict
    if i == 0:
        l_results = [{"prot_name": test_dataset[i][2], "context_latent": context_latent, "target_latent": target_latent, "full_seq": full_seq, "masked_seq": masked_seq}]
    else:
        l_results.append({"prot_name": test_dataset[i][2], "context_latent": context_latent, "target_latent": target_latent, "full_seq": full_seq, "masked_seq": masked_seq})


# Because cpu can't carry such a big tensor
# results = test_jepa(jepa, test_loader, device=torch.device('cpu'))

#save the dico as a .pt file
torch.save(l_results, 'latents/10k_SIGReg/cont_targ_lat_10k.pt')
print("Context and target latent representations saved to 'latents/10k_SIGReg/cont_targ_lat_10k.pt' respectively.")