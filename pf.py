#!/usr/bin/env python
# encoding: utf-8

"""
@brief Phrase finding with spark
@param fg_year The year taken as foreground
@param f_unigrams The file containing unigrams
@param f_bigrams The file containing bigrams
@param f_stopwords The file containing stop words
@param w_info Weight of informativeness
@param w_phrase Weight of phraseness
@param n_workers Number of workers
@param n_outputs Number of top bigrams in the output
"""

from __future__ import print_function
import sys
from pyspark import SparkConf, SparkContext
import math
from functools import partial

def preprocess(line, fg_year):
    tokens = line.split("\t")
    return (tokens[0], (int(tokens[2]), 0)) if int(tokens[1])==fg_year else (tokens[0], (0, int(tokens[2])))

def calc_phrAndinfo((first_word, second_word, bigram_FG, bigram_BG, first_FG, second_FG), \
                    unique_bigrams_count, unique_unigrams_count, total_bigrams_FG, total_bigrams_BG, \
                    total_unigrams_FG, w_phrase, w_info):

    # function for calculating kl divergence
    kld = lambda p,q : p * (math.log(p) - math.log(q))

    p_bigram_FG = float(bigram_FG + 1) / (unique_bigrams_count + total_bigrams_FG)
    p_bigram_BG = float(bigram_BG + 1) / (unique_bigrams_count + total_bigrams_BG)
    p_first_FG  = float(first_FG  + 1) / (unique_unigrams_count + total_unigrams_FG)
    p_second_FG = float(second_FG + 1) / (unique_unigrams_count + total_unigrams_FG)

    phraseness, informativeness = kld(p_bigram_FG, p_first_FG * p_second_FG), kld(p_bigram_FG, p_bigram_BG)

    return (first_word + "-" + second_word, w_phrase * phraseness + w_info * informativeness)

def main(argv):
    # parse args
    fg_year = int(argv[1])
    f_unigrams = argv[2]
    f_bigrams = argv[3]
    f_stopwords = argv[4]
    w_info = float(argv[5])
    w_phrase = float(argv[6])
    n_workers = int(argv[7])
    n_outputs = int(argv[8])

    """ configure pyspark """
    conf = SparkConf().setMaster('local[{}]'.format(n_workers))  \
                      .setAppName(argv[0])
    sc = SparkContext(conf=conf)

    # TODO: start your code here

    bigrams_raw = sc.textFile(f_bigrams)
    stop_words = set(sc.textFile(f_stopwords).collect())

    # Build partial function
    preprocess_p = partial(preprocess, fg_year=fg_year)

    bigram_tokens = bigrams_raw.map(preprocess_p).filter(lambda x : not reduce(lambda x, y : x or y, \
                                                    [word in stop_words for word in x[0].split(" ")]))

    aggregated_bigrams = bigram_tokens.reduceByKey(lambda x, y :(x[0] + y[0],x[1] + y[1]), n_workers)

    processed_bigrams = aggregated_bigrams.map(lambda x:tuple(j if i==0 else (j, x[1]) for i,j in \
                                                    enumerate(x[0].split(" ")))).persist()

    unique_bigrams_count = processed_bigrams.count()
    total_bigrams_FG, total_bigrams_BG = processed_bigrams.map(lambda x : (x[1][1][0], x[1][1][1])) \
                                                                .reduce(lambda x,y : (x[0]+y[0],x[1]+y[1]))

    unigrams_raw = sc.textFile(f_unigrams)
    unigram_tokens = unigrams_raw.map(preprocess_p).filter(lambda x : x[0] not in stop_words)

    processed_unigrams = unigram_tokens.reduceByKey(lambda x,y : (x[0]+y[0],x[1]+y[1]), n_workers).persist()

    unique_unigrams_count = processed_unigrams.count()
    total_unigrams_FG, _ = processed_unigrams.map(lambda x : (x[1][0], x[1][1])) \
                                             .reduce(lambda x, y : (x[0]+y[0], x[1]+y[1]))

    grouped_by_first_word = processed_unigrams.cogroup(processed_bigrams)
    bigrams_with_first_word_count = grouped_by_first_word.flatMap(lambda x: \
        map(lambda bigram: (bigram[0], (x[0], bigram[1][0], bigram[1][1], list(x[1][0])[0][0])),list(x[1][1])))

    grouped_by_second_word = processed_unigrams.cogroup(bigrams_with_first_word_count)

    bigrams_with_unigram_data = grouped_by_second_word.flatMap(lambda x: \
        map(lambda bigram : (bigram[0], x[0], bigram[1], bigram[2], bigram[3], list(x[1][0])[0][0]), list(x[1][1])))

    calc_phrAndinfo_p = partial(calc_phrAndinfo, unique_bigrams_count=unique_bigrams_count, \
                            unique_unigrams_count=unique_unigrams_count, total_bigrams_FG=total_bigrams_FG, \
                            total_bigrams_BG=total_bigrams_BG, total_unigrams_FG=total_unigrams_FG, \
                            w_phrase=w_phrase, w_info=w_info)

    scores_bigrams = bigrams_with_unigram_data.map(calc_phrAndinfo_p)

    # take the top K bigrams and print them to stdout
    map(lambda (bigram, total_score):print(bigram+":"+str(total_score)), scores_bigrams.takeOrdered(n_outputs, key=lambda x:-x[1]))    

    # print sc.parallelize(scores_bigrams.takeOrdered(n_outputs, key=lambda x:-x[1])).count()

    """ terminate """
    sc.stop()


if __name__ == '__main__':
    main(sys.argv)

