import math
from lm import UnigramLM,BigramLM,InterpolatedBigramModel
from utils import read_sentence, read_vocab, calculate_bigram_perplexity, calc_average_log_likelihood,calculate_bigram_document_perplexity,calc_average_document_log_likelihood
import numpy as np

# reading train/valid/test data
train_data = read_sentence('train.txt')
valid_data = read_sentence('valid.txt')
test_data = read_sentence('test.txt')
vocab = read_vocab('vocab.txt')

#run
print("--------------------")
print("Backoff Bigram Model")
print("--------------------")
bigram = BigramLM(train_data, vocab, smoothing=True)
print("Average log likelihood of first line in test: %s" % (calc_average_log_likelihood(bigram,test_data[:1])))
print("Average ppl of first line in test: %s" % (calculate_bigram_perplexity(bigram, test_data[:1])))
loglikelihood = calc_average_log_likelihood(bigram,test_data[:100])
ppl = calculate_bigram_perplexity(bigram, test_data[:100])
print("Mean of loglikelihood of first 100 line: %s" % np.mean(loglikelihood))
print("Variance of loglikelihood of first 100 line: %s" % np.var(loglikelihood))
print("Mean of ppl of first 100 line: %s" % np.mean(ppl))
print("Variance of ppl of first 100 line: %s" % np.var(ppl))
print("Average ppl of document: %s" % (calculate_bigram_document_perplexity(bigram,test_data)))
print("Average log likelihood of document: %s" % (calc_average_document_log_likelihood(bigram,test_data)))




print("\n")


print("-------------------------")
print("Interpolated Bigram Model")
print("-------------------------")
interpolated_bigram = InterpolatedBigramModel(train_data, vocab, lbda=[0.29627563745722774, 0.6385138402313691, 0.06521052231140334])
interpolated_bigram.em_train_lbda(valid_data)
print("lamda: %s" % interpolated_bigram.lbda)
print("Average log likelihood of first line in test: %s" % (calc_average_log_likelihood(interpolated_bigram,test_data[:1])))
print("Average ppl of first line in test: %s" % (calculate_bigram_perplexity(interpolated_bigram, test_data[:1])))
loglikelihood = calc_average_log_likelihood(interpolated_bigram,test_data[:100])
ppl = calculate_bigram_perplexity(interpolated_bigram, test_data[:100])
print("Mean of loglikelihood of first 100 line: %s" % np.mean(loglikelihood))
print("Variance of loglikelihood of first 100 line: %s" % np.var(loglikelihood))
print("Mean of ppl of first 100 line: %s" % np.mean(ppl))
print("Variance of ppl of first 100 line: %s" % np.var(ppl))
print("Average ppl of document: %s" % (calculate_bigram_document_perplexity(interpolated_bigram,test_data)))
print("Average log likelihood of document: %s" % (calc_average_document_log_likelihood(interpolated_bigram,test_data)))