from collections import defaultdict
import math
import re
import multiprocessing
class UnigramLM(object):
    def __init__(self, sentences, vocab, smoothing=False, alpha=0.01):
        self.vocab = set(vocab) # read frim vocab.txt
        self.vocab_size = len(self.vocab)
        self.corpus_length = 0 # calculate all the tokens in training data
        self.unigram_freq = defaultdict(lambda: 0)
        self.smoothing = smoothing
        self.alpha = alpha
        for sentence in sentences:
            for word in sentence:
                self.unigram_freq[word] += 1
                # whether word is in the given vocabulary
                if word in self.vocab:
                    self.corpus_length += 1

    def _cal_unigram_prob(self, word):
        numerator = self.unigram_freq[word]
        denumerator = self.corpus_length
        if self.smoothing:
            numerator += self.alpha
            denumerator += self.vocab_size*self.alpha

        return float(numerator) / float(denumerator)

    def _calculate_unigram_sentence_probability(self, sentence, normalize_probability=True):
        sentence_probability_log_sum = 0
        for word in sentence:
            word_probability = self._cal_unigram_prob(word)
            sentence_probability_log_sum += math.log(word_probability)
        return math.pow(math.e, sentence_probability_log_sum) if normalize_probability else sentence_probability_log_sum

    def evaluate(self, test_data, normalize_probability=True):
        return self._calculate_unigram_sentence_probability(test_data, normalize_probability=normalize_probability)


class BigramLM(UnigramLM):
    def __init__(self, sentences, vocab, smoothing=False, alpha=0.01, K=16):
        UnigramLM.__init__(self, sentences, vocab, smoothing=smoothing,alpha=alpha)
        self.threshold = K
        self.bigram_freq = defaultdict(lambda: 0)

        for sentence in sentences:
            prev = None
            for word in sentence:
                if prev != None:
                    self.bigram_freq[(prev, word)] += 1
                prev = word

        if self.smoothing:

            self.bigram_freq_count = defaultdict(lambda: 0)
            for value in self.bigram_freq.values():
                self.bigram_freq_count[value] += 1
                
            self.bigram_proba = defaultdict(lambda: 0)
            self._train(set(self.bigram_freq.keys()))

            self.prev_norm = defaultdict(lambda: 0)
            self.cal_norm(0,self.vocab)

    def cal_norm(self, i, vocab):
        for prev in vocab:
            if prev not in self.prev_norm:
                mass = 0
                backoff = 0
                s = set()
                for x in self.vocab:
                    if (prev, x) in self.bigram_proba:
                        s.add((prev, x))
                        mass += self.bigram_proba[(prev, x)]
                    else:
                        backoff += self._cal_unigram_prob(x)
                if mass >= 1.0:
                    for prev, x in s:
                        tmp = self.bigram_proba[(prev, x)]
                        self.bigram_proba[(prev, x)] = float(tmp)/float(mass)
                    self.prev_norm[prev] = 0.0
                else:
                    self.prev_norm[prev] = float(1 - mass) / float(backoff)

    def _train(self, keys):
        for prev, word in keys:
            bigram_word_probability = self._cal_bigram_probabilty(prev, word)
            self.bigram_proba[(prev, word)] = bigram_word_probability


    def _cal_bigram_probabilty(self, prev, word):
        r = self.bigram_freq[(prev, word)]
        numerator = 0.0
        denumerator = 0.0
        if self.smoothing:
            if 0 < r <= self.threshold:
                numerator = self.bigram_freq_count[r+1]*(r + 1)
                denumerator = self.bigram_freq_count[r]*self.unigram_freq[prev]
            elif r > self.threshold:
                numerator = r
                denumerator = self.unigram_freq[prev]
        else:
            numerator = self.bigram_freq[(prev, word)]
            denumerator = self.unigram_freq[prev]

        return 0.0 if numerator == 0 or denumerator == 0 else float(numerator) / float(denumerator)


    def _calculate_bigram_sentence_probability(self, sentence, normalize_probability=True):
        bigram_sentence_probability_log_sum = 0
        prev = None
        tmp = {}
        for word in sentence:
            # 1.prev == None 2. w_{t-1} not in training set => normlization is simply 1. => r == 0
            if prev not in self.vocab:
                bigram_word_probability = self._cal_unigram_prob(word) if self.smoothing else self._cal_bigram_probabilty(prev, word)
                bigram_sentence_probability_log_sum += math.log(bigram_word_probability) if bigram_word_probability != 0.0 else 0.0
            # calculate all (w_{t-1}, w_t) pair, and find (w_{t-1}, w_t) with 0 freq
            else:
                bigram_word_probability = self.bigram_proba[(prev, word)]
                if bigram_word_probability != 0.0:
                    bigram_sentence_probability_log_sum += math.log(bigram_word_probability)
                    tmp[(prev,word)] = bigram_word_probability
                else:
                    backoff = self._cal_unigram_prob(word)
                    norm_term = self.prev_norm[prev]
                    bigram_sentence_probability_log_sum += math.log(float(backoff) * float(norm_term)) if norm_term != 0.0 else 0.0
            prev = word

        return math.pow(math.e,
                        bigram_sentence_probability_log_sum) if normalize_probability else bigram_sentence_probability_log_sum

    def evaluate(self, test_data, normalize_probability=True):
        return self._calculate_bigram_sentence_probability(test_data, normalize_probability=normalize_probability)

class InterpolatedBigramModel(BigramLM):
    def __init__(self, sentences, vocab, lbda=[1/3.0, 1/3.0, 1/3.0] ,smoothing=False):
        BigramLM.__init__(self, sentences, vocab, smoothing=smoothing)
        self.lbda=lbda
        self.unigram_uniform = defaultdict(lambda: 0)
        for v in vocab:
            self.unigram_uniform[v] = float(1)/float(self.vocab_size)

    def _cal_unigram_uniform(self, word):
        return self.unigram_uniform[word]

    def _cal_interpolated_prob(self, prev, word):
        p_unigram_MLE = self.lbda[0]*self._cal_unigram_prob(word)
        if prev is None and prev not in self.vocab:
            p_bigram_MLE = self.lbda[1]*self._cal_unigram_prob(word)
        else:
            p_bigram_MLE = self.lbda[1]*self._cal_bigram_probabilty(prev, word)

        p_unigram_uniform = self.lbda[2]*self._cal_unigram_uniform(word)
        devisor = p_unigram_MLE + p_bigram_MLE + p_unigram_uniform
        return float(p_unigram_MLE), float(p_bigram_MLE), float(p_unigram_uniform), float(devisor)

    def em_train_lbda(self, validation_sentences, threshold=0.00001):
        iteration = 1
        pre_log, cur_log = 0.0, 0.0
        while True:
            N = 0
            sump = [0,0,0]
            for sentence in validation_sentences:
                prev = None
                for word in sentence:
                    p_unigram_MLE, p_bigram_MLE, p_unigram_uniform, devisor = self._cal_interpolated_prob(prev, word)
                    sump[0] += p_unigram_MLE / devisor
                    sump[1] += p_bigram_MLE / devisor
                    sump[2] += p_unigram_uniform / devisor
                    N += 1
                    prev = word
            self.lbda = [v/N for v in sump]
            print("")
            print("Iteration ", iteration)
            print("lambdas:", self.lbda)
            for sentence in validation_sentences:
                prev = None
                for word in sentence:
                    if prev != None:
                        _,_,_,likelihood = self._cal_interpolated_prob(prev, word)
                        cur_log += math.log(likelihood)
                    prev = word
            cur_log = cur_log / N
            likelihood_gain = (cur_log - pre_log) / abs(cur_log)
            if iteration != 1:
                print("log likelihood increased by the ratio of ", likelihood_gain)
                print("average log likelihood: ", cur_log)
            if iteration != 1 and likelihood_gain <= threshold:
                break
            if iteration != 1 and likelihood_gain < 0:
                print("wrong implementation")
            pre_log = cur_log
            iteration += 1

    def _calculate_interpolated_bigram_sentence_probability(self, sentence, normalize_probability=True):
        bigram_sentence_probability_log_sum = 0
        prev = None
        for word in sentence:
            _,_,_,bigram_word_probability = self._cal_interpolated_prob(prev, word)
            bigram_sentence_probability_log_sum += math.log(bigram_word_probability) if bigram_word_probability != 0.0 else 0.0
            prev = word
        return math.pow(math.e,
                        bigram_sentence_probability_log_sum) if normalize_probability else bigram_sentence_probability_log_sum

    def evaluate(self, test_data, normalize_probability=True):
        return self._calculate_interpolated_bigram_sentence_probability(test_data, normalize_probability=normalize_probability)