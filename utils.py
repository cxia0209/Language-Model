import re
import math
def read_sentence(filepath):
        with open(filepath, 'r') as f:
                return [re.split("\s+", line.rstrip('\n')) for line in f]

def read_vocab(filepath):
        with open(filepath, 'r') as f:
                return [line.rstrip('\n') for line in f]

def calculate_number_of_bigrams(sentence):
        return len(sentence)

def calculate_bigram_perplexity(model, sentences):
        ppl = []
        for sentence in sentences:
                number_of_bigrams = calculate_number_of_bigrams(sentence)
                bigram_sentence_probability_log_sum = -math.log(model.evaluate(sentence))
                ppl.append(math.pow(math.e, bigram_sentence_probability_log_sum / number_of_bigrams))
        return ppl

def calc_average_log_likelihood(model, sentences):
        avg_ll = []
        for sentence in sentences:
                avg_ll.append(math.log(model.evaluate(sentence))/len(sentence))
        return avg_ll

def calculate_bigram_document_perplexity(model, sentences):
        bigram_sentence_probability_log_sum = 0
        number_of_bigrams = 0
        for sentence in sentences:
                bigram_sentence_probability_log_sum -= math.log(model.evaluate(sentence))
                number_of_bigrams += calculate_number_of_bigrams(sentence)
        return math.pow(math.e, bigram_sentence_probability_log_sum / number_of_bigrams)

def calc_average_document_log_likelihood(model, sentences):
        bigram_sentence_probability_log_sum = 0
        number_of_bigrams = 0
        for sentence in sentences:
                bigram_sentence_probability_log_sum += math.log(model.evaluate(sentence))
                number_of_bigrams += calculate_number_of_bigrams(sentence)
        return bigram_sentence_probability_log_sum/ number_of_bigrams