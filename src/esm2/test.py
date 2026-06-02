from transformers import AutoModel, AutoTokenizer
import torch

print("tokenizer ...")

# Model tokenizer = dictionary of tokens and their corresponding IDs
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")

print("tokenizer loaded !")
# Print tokenizer information
vocab = tokenizer.get_vocab()
print(f"Taille totale du vocabulaire : {len(vocab)}")
print("Exemples de tokens :", dict(list(vocab.items())[:10]))


def mask_amino_acids(input_ids: torch.Tensor, tokenizer, mask_ratio: float = 0.2, seed: int = 42) -> torch.Tensor:
    """Mask a ratio of amino-acid tokens in a batch of token ids.

    Special tokens and padding are preserved.
    """
    masked_input_ids = input_ids.clone()
    generator = torch.Generator().manual_seed(seed)

    special_token_ids = {
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
    }

    for row_index in range(masked_input_ids.size(0)):
        valid_positions = [
            position
            for position, token_id in enumerate(masked_input_ids[row_index].tolist())
            if token_id not in special_token_ids
        ]

        if not valid_positions:
            continue

        num_to_mask = max(1, int(len(valid_positions) * mask_ratio))
        permutation = torch.randperm(len(valid_positions), generator=generator).tolist()
        positions_to_mask = [valid_positions[i] for i in permutation[:num_to_mask]]

        for position in positions_to_mask:
            masked_input_ids[row_index, position] = tokenizer.mask_token_id

    return masked_input_ids

# Test de tokenization
sequence = ["MVHFTAEEKAAVTSLWSKMNVEEAGGEALGRLLVVYPWTQRFFDSFGNLSSPSAILGNPKVKAHGKKVLTSFGDAIKNMDNLKPAFAKLSELHCDKLHVDPENFKLLGNVMVIILATHFGKEFTPEVQAAWQKLVSAVAIALAHKYH"]
tokens = tokenizer(sequence)
print(f"\nSéquence : {sequence}")
print(f"Tokens IDs : {tokens['input_ids']}")


model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
print("model loaded !")

inputs = tokenizer(sequence, return_tensors="pt", padding=True)
masked_input_ids = mask_amino_acids(inputs["input_ids"], tokenizer, mask_ratio=0.2, seed=42)
masked_inputs = dict(inputs)
masked_inputs["input_ids"] = masked_input_ids

print(f"\nTokens IDs masqués : {masked_input_ids}")


with torch.no_grad():
    outputs = model(**masked_inputs)

embeddings = outputs.last_hidden_state
print(f"Shape des inputs : {inputs['input_ids'].shape}") # [batch, sequence_length]
print(f"Shape des embeddings : {embeddings.shape}")     # [batch, sequence_length, hidden_dimension]

# Afficher les embeddings
print(f"\nEmbeddings de la séquence :")
print(embeddings)
print(type(embeddings))

# Sauvegarder les embeddings obtenus à partir de la séquence masquée
torch.save(embeddings, "embeddings.pt")
print(f"\nEmbeddings sauvegardés dans 'embeddings.pt'")
