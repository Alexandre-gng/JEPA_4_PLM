import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel, EsmConfig


def weak_sigreg_loss(x, sketch_dim=128):
    """
    Calcule la régularisation Weak-SIGReg pour un batch d'embeddings.
    Force l'espace latent à s'étaler sans s'effondrer.
    
    Args:
        x: Tensor [N, D] (ex: N = Batch * Seq, D = 1280)
        sketch_dim: Dimension de la projection aléatoire (généralement 64 ou 128)
    """
    B, D = x.shape
    
    # 1. Sketching (Projection aléatoire)
    # On passe de 1280 dimensions à sketch_dim sans paramètres à entraîner
    if D > sketch_dim:
        # Création d'une matrice aléatoire normalisée
        S = torch.randn(D, sketch_dim, device=x.device) / (D ** 0.5)
        x_proj = x @ S
    else:
        x_proj = x
        sketch_dim = D
        
    # 2. Centrage des données
    x_centered = x_proj - x_proj.mean(dim=0)
    
    # 3. Matrice de Covariance sur l'espace réduit (Taille: sketch_dim x sketch_dim)
    cov = (x_centered.T @ x_centered) / (B - 1)
    
    # 4. Pénalité : On compare à la Matrice Identité
    # (On veut des 1 sur la diagonale pour la variance, des 0 ailleurs pour l'indépendance)
    identity = torch.eye(sketch_dim, device=x.device)
    loss = F.mse_loss(cov, identity)
    
    return loss


def lejepa_loss(pred_x, target_y, lam=10.0, sketch_dim=128):
    """
    Loss complète type LeJEPA (Prediction + Régularisation Weak-SIGReg).
    """
    # 1. Invariance / Prediction Loss (Rapprochement du contexte prédit et de la cible)
    pred_loss = F.mse_loss(pred_x, target_y)
    
    # 2. Regularization Loss (Appliquée sur les cibles pour éviter l'effondrement de l'encodeur)
    reg_loss = weak_sigreg_loss(target_y, sketch_dim)
    
    # Total
    total_loss = pred_loss + lam * reg_loss
    
    return total_loss, pred_loss, reg_loss

# def test_lejepa_vram():
#     # --- CONFIGURATION ---
#     B = 4          # Réduit à 4 car on double les activations (2 passes avec gradients)
#     L_seq = 512
#     D = 1280
#     model_name = "facebook/esm2_t33_650M_UR50D"
    
#     print(f"Test LeJEPA (Weak-SIGReg) | Batch={B}, Seq={L_seq}")
    
#     # 1. Chargement
#     encoder = EsmModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
#     # Dans LeJEPA sans EMA, on entraîne tout le monde
#     encoder.train() 
    
#     # 2. Inputs
#     ids_context = torch.randint(0, 33, (B, L_seq), device='cuda')
#     ids_target = torch.randint(0, 33, (B, L_seq), device='cuda')
    
#     torch.cuda.reset_peak_memory_stats()
    
#     # 3. Double Forward (AVEC GRADIENTS pour les deux)
#     # Note: Dans un vrai projet, le Predictor transformerait context_out en pred_out
#     target_out = encoder(ids_target).last_hidden_state
#     context_out = encoder(ids_context).last_hidden_state
    
#     # Aplatissement pour la loss
#     target_flat = target_out.view(-1, D).float()
#     pred_flat = context_out.view(-1, D).float()
    
#     # 4. Loss
#     loss, p_loss, r_loss = lejepa_loss(pred_flat, target_flat)
    
#     print(f"VRAM avant Backward: {torch.cuda.memory_allocated()/(1024**2):.2f} MB")
    
#     # 5. Backward
#     loss.backward()
    
#     peak = torch.cuda.max_memory_allocated()/(1024**2)
#     print(f"VRAM MAX (Backward inclus): {peak:.2f} MB")
#     if peak > 15000:
#         print("⚠️ Proche de la limite des 16Go de la 4090 Laptop !")

# if __name__ == "__main__":
#     test_lejepa_vram()