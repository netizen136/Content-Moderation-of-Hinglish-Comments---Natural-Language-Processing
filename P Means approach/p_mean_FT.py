import numpy as np
from sklearn.preprocessing import normalize

def z_norm(vector):
    if(np.sum(vector) == .0):
        return vector
    return (vector-np.mean(vector))/np.std(vector)

def gen_mean(vals, p):
    p = float(p)
    return np.power(
        np.mean(
            np.power(
                np.array(vals, dtype=complex),
                p),
            axis=0),
        1 / p
    )


operations = dict([
    ('mean', (lambda word_embeddings: [np.mean(word_embeddings, axis=0)], lambda embeddings_size: embeddings_size)),
    ('max', (lambda word_embeddings: [np.max(word_embeddings, axis=0)], lambda embeddings_size: embeddings_size)),
    ('min', (lambda word_embeddings: [np.min(word_embeddings, axis=0)], lambda embeddings_size: embeddings_size)),
    ('p_mean_2', (lambda word_embeddings: [gen_mean(word_embeddings, p=2.0).real], lambda embeddings_size: embeddings_size)),
    ('p_mean_3', (lambda word_embeddings: [gen_mean(word_embeddings, p=3.0).real], lambda embeddings_size: embeddings_size)),
    ('p_mean_4', (lambda word_embeddings: [gen_mean(word_embeddings, p=4.0).real], lambda embeddings_size: embeddings_size)),
    ('hm', (lambda word_embeddings: [gen_mean(word_embeddings, p= -1).real], lambda embeddings_size: embeddings_size)),
])


def get_sentence_embedding(sentence, embeddings, chosen_operations):
    word_embeddings = []
    for tok in sentence:
        vec = embeddings.get_word_vector(tok)
        word_embeddings.append(np.squeeze(normalize(vec.reshape(1,-1))))

    if not word_embeddings:
        print('No word embeddings for sentence:\n{}'.format(sentence))
        size = 0
        for o in chosen_operations:
            size += operations[o][1](np.embeddings.get_dimension())
        sentence_embedding = np.zeros(size)
    else:
        concat_embs = []
        for o in chosen_operations:
            concat_embs += operations[o][0](word_embeddings)
        sentence_embedding = np.concatenate(
            concat_embs,
            axis=0
        )

    return z_norm(sentence_embedding)
