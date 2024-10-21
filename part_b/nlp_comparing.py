import pandas as pd

def compare_docs(**kwargs):
    df = []
    for arg in kwargs.values():
        sub = []
        for arg2 in kwargs.values():
            sub.append(arg.similarity(arg2))
        df.append(sub)
    return pd.DataFrame(df, columns=list(kwargs.keys()), index=list(kwargs.keys()))


def compare_tokens(doc1, doc2, token_count=100):
    words1 = []
    words2 = []
    value = []
    for i in range(token_count):
        words1.append(doc1[i])
        words2.append(doc2[i])
        value.append(doc1[i].similarity(doc2[i]))
    return pd.DataFrame(data=[words1, words2, value], index=['Token 1', 'Token 2', 'Similarity']).T