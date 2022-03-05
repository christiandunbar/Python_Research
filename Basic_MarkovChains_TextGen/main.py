import numpy as np


def generateTable(data, k=4):
    t = {}
    for i in range(len(data) - k):
        x = data[i:i + k]
        y = data[i + k]

        if t.get(x) is None:
            t[x] = {}
            t[x][y] = 1
        else:
            if t[x].get(y) is None:
                t[x][y] = 1
            else:
                t[x][y] += 1

    return t


def convertFreqIntoProb(T):
    for kx in T.keys():
        s = float(sum(T[kx].values()))
        for k in T[kx].keys():
            T[kx][k] = T[kx][k] / s

    return T


text_path = "train_corpus.txt"


def load_text(filename):
    with open(filename,encoding='utf8') as f:
        return f.read().lower()


text = load_text(text_path)
print('Loaded the dataset.')


def MarkovChain(text, k=4):
    t = generateTable(text, k)
    t = convertFreqIntoProb(t)
    return t

model = MarkovChain(text)
print('Model created successfully')


def sample_next(ctx, model, k):
    ctx = ctx[-k:]
    if model.get(ctx) is None:
        return " "
    possible_Chars = list(model[ctx].keys())
    possible_values = list(model[ctx].values())

    print(possible_Chars)
    print(possible_values)

    return np.random.choice(possible_Chars, p=possible_values)


def generateText(starting_sent, k=4, maxLen=1000):

    sentence = starting_sent
    ctx = starting_sent[-k:]

    for ix in range(maxLen):
        next_prediction = sample_next(ctx, model, k)
        sentence += next_prediction
        ctx = sentence[-k:]
    return sentence


print("Function created successfully")


text = generateText("dear", k=4, maxLen=2000)
print(text)