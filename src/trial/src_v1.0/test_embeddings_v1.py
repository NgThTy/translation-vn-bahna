import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from train_embeddings import BahVnPairsDataset, collate_fn, build_models, evaluate

# --- paths and params ---
test_csv = "../data/test.csv"
output_dir = "../results/models_demo"
src_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tgt_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- load dataset and tokenizer ---
test_ds = BahVnPairsDataset(test_csv)
src_tokenizer = AutoTokenizer.from_pretrained(src_model_name)
tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_model_name)
test_loader = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False,
    collate_fn=lambda b: collate_fn(
        b, src_tokenizer, tgt_tokenizer, 16, 16, device
    ),
)

# --- rebuild model architecture and load weights ---
src_model, tgt_model = build_models(src_model_name, tgt_model_name, 256, freeze_base=False, device=device)
src_model.proj.load_state_dict(torch.load(f"{output_dir}/src_proj.pt", map_location=device))
tgt_model.proj.load_state_dict(torch.load(f"{output_dir}/tgt_proj.pt", map_location=device))

# --- evaluate ---
test_loss = evaluate(src_model, tgt_model, test_loader, device)
print(f"Test MSE loss: {test_loss:.6f}")