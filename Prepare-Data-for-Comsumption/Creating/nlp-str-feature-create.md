
## nlp string create feature

```python
import pandas as pd

# 1 create one feature
df['q1len'] = df['question1'].str.len()
df['q2len'] = df['question2'].str.len()

# 2 create two feature
df['q1_n_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
df['q2_n_words'] = df['question2'].apply(lambda row: len(row.split(" ")))


# 3 create three feature
def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(' ')))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(' ')))
    return 1.0 * len(w1 & w2) / (len(w1) + len(w2))

df['word_share'] = df.apply(normalized_word_share, axis=1)
```

