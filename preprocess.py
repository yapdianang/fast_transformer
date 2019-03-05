import os
import sys
from string import punctuation
from tqdm import tqdm

def readFile(path):
	summary = []
	article = []
	lines = []
	with open(path, 'r') as f:
		lines = [x.strip() for x in f.readlines()]

	summ = True
	for line in lines:
		if line == "@summary" or line == "":
			continue
		elif line == "@article":
			summ = False
		elif summ:
			summary.append(line.strip(punctuation))
		else:
			article.append(line)

	summary = ", ".join(summary) + "."
	article = " ".join(article)
	return summary, article

def preprocess(dir):
	f = "data/all_{}.txt".format(dir)
	lines = []
	with open(f, 'r') as f:
		lines = [x.strip() for x in f.readlines()]

	articles = []
	summaries = []

	for name in tqdm(lines):
		path = "data/articles/{}.txt".format(name)
		summary, article = readFile(path)
		summaries.append(summary)
		articles.append(article)

	with open("data/{}/summaries.txt".format(dir), 'w') as f:
		count = 0
		for summ in summaries: # Write piecemail to avoid high memory usage
			count += 1
			if count == len(summaries):
				f.write('%s' % summ)
			else:
				f.write('%s\n' % summ)

	with open("data/{}/articles.txt".format(dir), 'w') as f:
		count = 0
		for art in articles: # Write piecemail to avoid high memory usage
			count += 1
			if count == len(articles):
				f.write('%s' % art)
			else:
				f.write('%s\n' % art)

if __name__ == '__main__':
	preprocess("val")