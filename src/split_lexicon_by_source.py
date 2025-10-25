# split_lexicon_by_source.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

lex_path = "../data/lexicon.csv"
out_train = "../data/lexicon_train.csv"
out_test  = "../data/lexicon_test.csv"
test_size = 0.2
seed = 42

df = pd.read_csv(lex_path)
df = df.dropna(subset=["Bahnaric","Vietnamese"]).drop_duplicates()

# unique Bahnaric words
src_words = sorted(df["Bahnaric"].astype(str).unique())

# split the set of source words
src_train, src_test = train_test_split(src_words, test_size=test_size, random_state=seed)

src_train = set(src_train)
src_test  = set(src_test)

# filter pairs: all pairs whose Bahnaric in src_train go to train; src_test to test
train_df = df[df["Bahnaric"].isin(src_train)].copy()
test_df  = df[df["Bahnaric"].isin(src_test)].copy()

# save
train_df.to_csv(out_train, index=False)
test_df.to_csv(out_test, index=False)

print(f"Saved {len(train_df)} train rows ({len(src_train)} Bahnaric types)")
print(f"Saved {len(test_df)}  test rows ({len(src_test)} Bahnaric types)")
