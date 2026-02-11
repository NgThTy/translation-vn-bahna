# Bahnaric - Vietnamse: Projection, Alignment & Sentence Translation (Windows, Local)

This repository fine-tunes lightweight multilingual sentence models with small projection heads to align Bahnaric and Vietnamese embeddings. It then performs a supervised Procrustes (Kabsch) alignment for word-level vocabularies and evaluates sentence-level translation via retrieval with cosine/CSLS and topk_accuracy/BLEU/chrF.
- Target OS: Windows 10/11 
- Python: 3.12.7
- Hardware: Local machine (CPU works; CUDA will be used automatically if available)

## Repository layout
- data/
    - train.csv
    - test. csv
    - lexicon.csv
    - src_vocab.csv
    - tgt_vocab.csv
- results/
    - models_demo/
    - alignment/
    - sent_eval/
- src/
    - train_embeddings.py
    - test_embeddings.py
    - split_lexicon_by_source.py
    - align_embeddings.py
    - translate_and_eval_sentences.py
- requirements.txt
- README.md
- .gitignore

## Data formats
data/train.csv, data/test.csv (sentences) \
Required columns:
- Bahnaric : string
- Vietnamese: string 

data/lexicon.csv (word pairs for alignment/eval) \
Required columns:
- Bahnaric : token 
- Vietnamese: token

data/src_vocab.csv, data/tgt_vocab.csv (vocabularies) \
Required columns:
- word : token string (vocabulary item)
- text : surface form/surrounding text used for embedding

## Windows setup
1. Install Python 3.12.7 (add “Add python.exe to PATH”). 
2. Open PowerShell in the repo root.
3. Create & activate a virtual environment \
`python -m venv .venv` \
`.\.venv\Scripts\Activate.ps1`
4. Upgrade pip & install dependencies \
`python -m pip install --upgrade pip`\
`pip install -r requirements.txt`


## Run from repo root
This runs: split lexicon → train projection heads (sentences) → align on words → translate & evaluate sentences.

0) Ensure UTF-8 console \
`$Env:PYTHONIOENCODING="utf-8" `

1) Split lexicon into train/test \
`python .\src\split_lexicon_by_source.py`

2) Train projection heads on sentence pairs \
`python .\src\train_embeddings.py `\
`  --train_csv .\data\train.csv `\
`  --src_model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `\
`  --tgt_model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `\
`  --projection_dim 256 `\
`  --freeze_base `\
`  --epochs 1 --batch_size 8 --lr 2e-4 `\
`  --src_max_len 256 --tgt_max_len 256 `\
`  --output_dir .\results\models_demo `\
`  --num_workers 0 --pin_memory`

3) Build vocab embeddings, learn R,t with Kabsch, evaluate on held-out lexicon \
`python .\src\align_embeddings.py `\
`  --src_emb_csv .\data\src_vocab.csv `\
`  --tgt_emb_csv .\data\tgt_vocab.csv `\
`  --align_pairs_csv .\data\lexicon_train.csv `\
`  --eval_pairs_csv .\data\lexicon_test.csv `\
`  --proj_dir .\results\models_demo `\
`  --output_dir .\results\alignment `\
`  --topk 5 `\
`  --src_max_len 16 --tgt_max_len 16 `

4) Sentence translation via retrieval + BLEU/chrF \
`python .\src\translate_and_eval_sentences.py `\
`  --test_csv .\data\test.csv `\
` --proj_dir .\results\models_demo `\
` --alignment_dir .\results\alignment `\
`  --src_model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `\
`  --tgt_model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `\
`  --src_max_len 256 --tgt_max_len 256 `\
`  --batch_size 8 `\
`  --use_idf_pool `\
`  --use_csls `\
`  --csls_k 10 `\
`  --output_dir .\results\sent_eval `

### Outputs to verify
- results\models_demo\src_proj.pt, tgt_proj.pt, meta.txt
- results\alignment\R.npy, t.npy, top_predictions_sample.csv
- results\sent_eval\sentence_predictions.csv