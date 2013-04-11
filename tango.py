from __future__ import division
from collections import defaultdict
import glob
import os
import re
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import operator
from optparse import OptionParser
import numpy as np

# first generate all ngram counts in the entire corpus
def ngramsCount(filename, N):
	'''
	function that generates all ngram counts in the corpus
		args: filename of the alice_nospaces.txt file
		N: extract all ngrams of size 1 to N
	returns: 
		tokensdict: dictionary of ngram coumts
	'''
	# initalize dictionary
	tokensdict = defaultdict(lambda:0)

	# open file
	f = open(filename, 'rt')

	# iterate through lines
	for line in f.readlines():
		tokens = list(line.strip())
		n_tokens = len(tokens)
		# calculate all ngrams
		for i in range(n_tokens):
			begin = i+1
			end = min(n_tokens, i+N)+1

			for j in range(begin, end):
				tokenid = tokens[i:j]
				tokenid = ''.join(tokenid)

				# enter into dictionary
				tokensdict[tokenid] += 1
	f.close()
	return tokensdict

def removeSpaces(input_filename, output_filename):
	'''
	function that removes spaces from the alice_tokenized text
		args: 
			input_filename: name of alice_tokenized.txt
			output_filename: name of alice_nospaces.txt
		returns: 
			nothing
	'''
	f = open(input_filename, 'r')
	textContents = f.read()
	newTextContents = textContents.replace(' ','')
	g = open(output_filename, 'w')
	g.write(newTextContents)
	g.close()
	f.close()

def calculateNgramPossible(sentence, n, k, tokensdict):
	'''
	helper function to tango that computes straddle vs. left right
		args: sentence, the sentence of interest
		n: size of n-gram
		k: location
		tokensdict: the dictionary of ngrams
	returns normalized_vote, the normalized vote for fixed n and k

	'''
	vote = 0
	# creating padding
	buf = []

	# padding the sentence
	for i in range(0,n-1):
		buf.append('#')
	paddedSentence = buf + sentence + buf
	nk = k + (n-1)
	# n-gram on either side of potential split
	lng = ''.join(paddedSentence[(nk-n):nk])
	rng = ''.join(paddedSentence[nk:(nk+n)])
	# n-grams that straddle the potential split, there are n-1 of them
	for i in xrange((nk-n+1),nk):
		j = i+n
		straddleng = ''.join(paddedSentence[i:j])
		# check how often these straddling ngrams appeared in the corpus
		if tokensdict[lng] > tokensdict[straddleng]:
			vote += 1
		if tokensdict[rng] > tokensdict[straddleng]:
			vote += 1
	normalized_vote = vote * (1/(2*(n-1)))
	return normalized_vote

def tango(sentence, N, t, tokensdict):
	''' 
	function that performs tango algorithm
		args: sentence - text string of interest
			N - order of ngrams to calculate, from 1 to N
			tokensdict - dictionary of all ngrams 1 to N in the corpus
		returns segmentedsentence - the segmented sentence
			'''
	# take sentence and determine all possible locations to split, from the left
	sentence_length=len(sentence)
	
	# fixing k
	location_vector = defaultdict(lambda:0)
	for k in xrange(1, sentence_length):
		# fixing n
		vote_vector = []
		for n in xrange(2,N+1):
			vote = calculateNgramPossible(sentence, n, k, tokensdict)
			vote_vector.append(vote)
		# calculating the total vote
		total_vote = sum(vote_vector)/N
		location_vector[k] = total_vote

	# boundaries are now placed at all locations l such that 
	whereToPutSpaces = []
	for k in xrange(1,sentence_length):
		if (location_vector[k] > location_vector[k+1] and location_vector[k] > location_vector[k-1]):
			whereToPutSpaces.append(k)
		elif location_vector[k] > t:
			whereToPutSpaces.append(k)

	#  include the last slice
	whereToPutSpaces.append(len(sentence)) 
	#  start with first slice
	temp = 0  
	result = []
	for i in whereToPutSpaces:
  		result.append(sentence[temp:i])
  		temp = i

  	prelimsegmentedsentence = ["".join(x) for x in result]
  	segmentedsentence = ' '.join(prelimsegmentedsentence)
	return [segmentedsentence,whereToPutSpaces]

def evaluate(segmentedBlock,goldStandard):
	'''
	function that takes in blocks of segmented and gold standard text and calculates precision and recall
		args: segmentedBlock - block of segmented text with tango
			goldStandard - block of goldStandard text
		returns: [precision, recall] - a list with elements precision and recall
	'''
	segmentedIndices = segmentedBlock
	goldIndices = goldStandard

	# calculate tp, fp, fn using set notation
	truePositives = len(set(goldIndices).intersection(segmentedIndices))
	falsePositives = len(set(segmentedIndices) - set(goldIndices))
	falseNegatives = len(set(goldIndices) - set(segmentedIndices))

	precision = truePositives/(truePositives+falsePositives)
	
	recall = truePositives/(truePositives + falseNegatives)
	return [precision, recall]

def findGoldStandardSpaces(gl):
	'''
	function that calculates the correct indices for the gold standard
		args: takes in gl, a line of text
	returns: list of indices of spaces in the original goldstandard text
	'''
	# indices of blank spaces of gl turned into list
	indices = [i for i, x in enumerate(list(gl)) if x == ' ']

	# indices adjusted for shifting
	indices_correct = [x-i for i,x in enumerate(indices)]
	return indices_correct

def runWholeTango(unsegmentedFilename,noSpacesFilename,newSegmentedFilename, norder, t):
	'''
	function that runs all of tango_result
	args: 
		unsegmentedFilename: original goldStandard
		noSpacesFilename: original with no spaces
		newSegmentedFilename:
		norder: number of ngrams to iterate through
		t: threshold for tango
	returns:
		metric: list of precision and recall values
	''' 

	# open gold standard
	d = open(unsegmentedFilename, 'rt')
	goldStandard = d.readlines()

	# remove spaces
	removeSpaces(unsegmentedFilename,noSpacesFilename)

	# populate dictionary
	tokensdict = ngramsCount(noSpacesFilename, norder)

	# open input file
	f = open(noSpacesFilename, 'rt')
	textContents = f.readlines()

	# open outputfile
	g = open(newSegmentedFilename, 'w+')

	# lists for checking precision and recall
	segmentedSpaces = []
	goldStandardSpaces = []

	m = 0
	for line in textContents:
		ll = list(line.strip())
		gl = goldStandard[m].strip()

		# check gold standard spaces
		goldStandardSpaces += findGoldStandardSpaces(gl)
		tango_result = tango(ll,norder,t, tokensdict)

		# segmented text after tango
		segmented_text = tango_result[0]

		# spaces that they segmented
		segmentedSpaces += tango_result[1]

		# write to output file
		g.write(segmented_text+'\n')

		m += 1
	g.close()
	p = open(newSegmentedFilename, 'r+')
	segmentedBlock = p.read()
	metric = evaluate(segmentedSpaces, goldStandardSpaces)
	return metric

def main():
	'''
	main function deals with parsing command line arguments
		args:
		input_filename: the gold standard  (string)
		segmented_filename: the output file (string)
		threshold: the threshold for tango (float)
		sizengrams: size of n-grams to iterate through (n)
		g: boolean flag for whether we want to the graph to be generated
	'''
	parser = OptionParser()
	parser.add_option("-f", "--input_filename", default = 'alice_tokenized.txt', help="input file")
	parser.add_option("-p", "--segmented_filename", default = 'alice_segmented.txt', help='segmented filename')
	parser.add_option("-t", "--threshold", default = .95, help = "threshold for tango")
	parser.add_option("-n", "--sizengrams", default = 10, help = "size of ngrams")
	parser.add_option("-g", action="store_true", dest="verbose")

	(options, args) = parser.parse_args()
	unsegmentedFilename = options.input_filename
	segmentedFilename = options.segmented_filename
	newsegmentedFilename = 'alice_newsegmented.txt'
	t = options.threshold
	norder = int(options.sizengrams)

	# set no spaces filename
	noSpacesFilename = 'alice_nospaces.txt'

	metric = runWholeTango(unsegmentedFilename,noSpacesFilename,segmentedFilename, norder, t)

	print('precision: '+str(metric[0]))
	print('recall: '+str(metric[1]))

	if options.verbose == True:
		graph_precision_vector = []
		graph_recall_vector = []
		for threshold in range(0,100,5):
			thresh = threshold/100
			graph_metric = runWholeTango(unsegmentedFilename,noSpacesFilename,newsegmentedFilename, norder, thresh)
			graph_precision_vector.append(graph_metric[0])
			graph_recall_vector.append(graph_metric[1])

		plt.clf()
		p1 = plt.plot(graph_precision_vector, graph_recall_vector, color='b')
		plt.title('precision vs recall')
		plt.xlabel('precision')
		plt.ylabel('recall')
		plt.show()


if __name__ == "__main__":
    main()
