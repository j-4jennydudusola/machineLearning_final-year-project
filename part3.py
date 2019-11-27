import nltk
def process_file(file):
    with open(file, 'r') as f:
        sentences = []
        for line in f.readlines():
            sentence = line.split(' ')
            if len(sentence) >= 2 and sentence[0] == '<s>' and sentence[-1] == '</s>\n':
                sentence[-1] = sentence[-1].strip('\n')
                sentences.append(sentence)
            else:
                print("sentence not added")
    return sentences


def read_vocabulary(vocab_file):
    with open(vocab_file, "r") as f:
        return [line.rstrip("\n") for line in f.readlines()]


def compute_unigram(file, vocab, smoothing):
    unigrams = {}
    N = 0
    unigram_count = len(vocab) + 1

    for word_type in vocab:
        unigrams[word_type] = 0
    unigrams["UNK"] = 0
    for sentence in file:
        for token in sentence:
            if token == "<s>" or token == "</s>":
                continue
            if token not in vocab:
                token = "UNK"
                unigrams[token] += 1
            else:
                unigrams[token] += 1
            N += 1



    # calculate probability for each word type in vocabulary
    for unigram in unigrams.keys():
        unigrams[unigram] = (unigrams[unigram] + smoothing) / (N + (unigram_count * smoothing))

    return unigrams



def print_unigram_probabilities(unigram_dictionary):
    for u in unigram_dictionary.keys():
        print(u + ":" + str(unigram_dictionary[u]))



def compute_bigram(text, vocab, smoothing):

    token_list = []
    for sentence in text:
        for word in sentence:
            token_list.append(word)

    list_bigramns = list(nltk.bigrams(token_list))
    bigram = {}

    for tuple in list_bigramns:
        if(tuple[0] == "</s>"):
            continue
        else:
            bigram[tuple] = 0

    for t in list_bigramns:
        if t in bigram.keys():
            bigram[t] += 1


    unigram_dictionary = {}
    bigram_p = dict.fromkeys(bigram, 0)

    b_vocab = list(vocab)
    b_vocab.append("</s>")
    b_vocab.append("<s>")

    for word in b_vocab:
        unigram_dictionary[word] = 0

    for sentence in text:
        for token in sentence:
            if token in b_vocab:
                unigram_dictionary[token] += 1
            else:
                unigram_dictionary["UNK"] += 1

    print(bigram)
    # calculate probability of each tuple
    for t in bigram:
        wordbefore = t[0]
        bigram_p[t] = (bigram[t] + smoothing) / (unigram_dictionary[wordbefore] + (len(b_vocab) * smoothing))


    return bigram_p

#
def sentence_probabilities(text, unigram, bigram):

    trigram_uni_prob = {}
    trigram_bi_prob = {}

    for line in text:
        sentence = tuple(line)
        trigram_uni_prob[sentence] = 1
        trigram_bi_prob[sentence] = 1

    unigram_s = unigram
    bigram_s = bigram

    for sent in trigram_uni_prob:
        for probability in unigram_s:
            trigram_uni_prob[sent] *= unigram_s[probability]


    for sent in trigram_bi_prob:
        for probability in bigram_s:
            trigram_bi_prob[sent] *= bigram_s[probability]

    print(trigram_uni_prob)
    print(trigram_bi_prob)


def print_bigram_probabilities(bigram_dictionary):
    for value in bigram_dictionary.keys():
        print(value + ":" + str(bigram_dictionary[value]))



def main():
    data_file = './sampledata.txt'
    vocab_file = './sampledata.vocab.txt'
    test_file = './sampletest.txt'
    test_vocab = './train.vocab.txt'

    data = process_file(data_file)
    vocabulary = read_vocabulary(vocab_file)
    test = process_file(test_file)

    test_vocabulary = read_vocabulary(test_vocab)

    print("----------- Toy dataset --------------")
    print("====UNIGRAM MODEL====")
    print("- Unsmoothed -")
    print_unigram_probabilities(compute_unigram(data, vocabulary, 0))
    print()
    print("- Smoothed -")
    print_unigram_probabilities(compute_unigram(data, vocabulary, 1))
    print()
    print("====BIGRAM MODEL====")
    print("- Unsmoothed -")
    print(compute_bigram(data, compute_unigram(data, vocabulary, 0), 0))
    print("- Smoothed -")
    #print(compute_bigram(data, vocabulary, 1))
    print("====TRIGRAM MODEL====")
    #print(sentence_probabilities(test, compute_unigram(test, vocabulary, 1),
                                 #compute_bigram(test, compute_unigram(test, vocabulary, 1), 1)))




if __name__ == "__main__":
    main()
