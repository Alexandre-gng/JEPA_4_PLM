from JEPA.JEPA import JEPA

from dataset.data_func import mask_sequence

from JEPA.train_JEPA import train_jepa, compute_loss


jepa = JEPA(latent_dim=320, output_dim=320, tau=0.99)
print("JEPA model initialized !")

# Loading dataset
dataset_path = "src/dataset/UniRef50/uniref50_half.fasta"
with open(dataset_path, "r") as f:
    sequences = []
    current_sequence = ""
    for line in f:
        if line.startswith(">"):
            if current_sequence:
                sequences.append(current_sequence)
                current_sequence = ""
        else:
            current_sequence += line.strip()
    if current_sequence:
        sequences.append(current_sequence)


masked_sequences = [mask_sequence(seq) for seq in sequences]
print(f"Nombre de séquences masquées : {len(masked_sequences)}")
print(f"Exemple de séquence masquée : {masked_sequences[0]}")

print(f"Nombre de séquences chargées : {len(sequences)}")


context_latent, target_latent = train_jepa(jepa, sequences[0], masked_sequences[0])

print("JEPA forward pass completed !")
print(f"Context latent shape: {context_latent.shape}")
print(f"Target latent shape: {target_latent.shape}")
print(f"Context latent example: {context_latent[0][:5]}")
print(f"Target latent example: {target_latent[0][:5]}")

from regularization.VICReg import VICRegLoss

computed_loss = compute_loss(VICRegLoss(), context_latent, target_latent)
print(f"Computed loss: {computed_loss}")



