
import collections
import re
import sys
from collections import Counter
from enum import Enum
from os import listdir
from os.path import isfile, join

from wordfreq import word_frequency, zipf_frequency

import nltk

# nltk setup, to be able to use the tokenizer.
nltk.download('punkt')
nltk.download('stopwords')

# Consts and enums
RELATIVE_INPUT_SOURCE_DIR = "books/"
DEBUG_MODE = False
# This is the ratio to use if a legitimate english word appears in the document that appears ~0 times in English normally.
RARE_WORD_RELATIVE_MULTIPLIER = 1000
TOKEN_BLACKLIST = [".", ",", "?", "!", "..."]
# We throw out words in the query with a zipf score above this value.
ZIPF_FREQUENCY_THRESHOLD = 6
STOPWORDS = nltk.corpus.stopwords.words("english")


class Capabilities(Enum):
    # Simple number of times that the word appears in the document source.
    SIMPLE_WORD_FREQ = 1
    # The ratio between the appearance rate in the doc, vs. in all English text.
    RELATIVE_WORD_FREQ = 2
    # NLTK provided frequency distribution
    NLTK_WORD_FREQ = 3


class Flags(Enum):
    # Use context_based scoring over other kinds of scoring
    CONTEXT_BASED_SCORING = 1


class Utils:
    def naive_tokenize(string):
        """
        Naive implementation of getting tokens out of a string.
        """
        return [word.strip() for word in string.split(" ")]

    def better_tokenize(string):
        """
        Smarter tokenizing from NLTK. We tokenize using the library, then filter punctuation and contractions, and make everything lowercase.
        """
        return list(map(lambda w: w.lower(), filter(lambda word: not("'" in word or word in TOKEN_BLACKLIST), nltk.word_tokenize(string))))

    def best_tokenize(string):
        """
        Best tokenizing, using NLTK, and then programatically avoiding need for a blacklist. Filtering out common words via NLTK's given blacklist.
        """
        words = [w.lower() for w in nltk.word_tokenize(string)
                 if w.isalpha() and w.lower() not in STOPWORDS]
        return words

    def score_ngrams(book_ngram, query_ngram):
        """
        We compare the similarity of a query ngram to a book ngram by seeing how
        frequently ngrams from the query appear in the book. We normalize the
        book frequency by diving by the number of words in the book, to avoid
        penalizing short books. Multiply by 100 to get a larger value.
        """
        book_num_words = sum(book_ngram.values())
        score = 0.0
        for ngram in query_ngram:
            if book_ngram[ngram] > 1:
                print("ngram \"{}\" appears {} times in query, {} times in book".format(
                    ngram, query_ngram[ngram], book_ngram[ngram]))
            score += query_ngram[ngram] * \
                (book_ngram[ngram] / book_num_words) * 100000
        return score

    def fprint(text):
        original_stdout = sys.stdout
        with open("debug_log.txt", 'a+') as debug_file:
            sys.stdout = debug_file
            print(text)
            sys.stdout = original_stdout


def build_document(title, raw_text):
    """
    Note: raw_text is a list of lines in the book.

    We hide the Document class so it can only be instantiated via the builder.
    """
    class Document:
        def __init__(self, title):
            self.title = title
            self.capabilities_list = []

        def __repr__(self):
            return "\"{}\"".format(self.title)

        def affinity_score(self, raw_query):
            def process_query(raw_query):
                # First, filter out common words from the query.
                # zipf frequencies are effectively on a scale of 0 to 8. They
                # correspond to the base-10 logarithm of the number of times the
                # word appears per billion words.
                # See https://pypi.org/project/wordfreq/ for details.
                words = Utils.best_tokenize(raw_query)
                words = filter(lambda word: zipf_frequency(
                    word, "en") < ZIPF_FREQUENCY_THRESHOLD, words)
                return words
            total_score = 0.0
            query_words = process_query(raw_query)

            query_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(
                query_words)
            query_trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(
                query_words)
            bigram_score = Utils.score_ngrams(
                self.nltk_bigram_finder.ngram_fd, query_bigram_finder.ngram_fd)
            trigram_score = Utils.score_ngrams(
                self.nltk_trigram_finder.ngram_fd, query_trigram_finder.ngram_fd)

            Utils.fprint("ngram scores for Book {}: 2: {}, 3: {}".format(
                self.title, bigram_score, trigram_score))
            total_score += bigram_score + trigram_score

            words_counter = Counter(query_words)
            original_stdout = sys.stdout
            with open("debug_log.txt", 'a+') as debug_file:
                sys.stdout = debug_file
                # For debugging purposes, let's print out exactly how we are calculating our scores.
                print("Scoring Book {} for query {}...:".format(
                    self.title, raw_query[:10]))
                print("--------------------------------")

                # List of (query_word, frequency, k_value, point_value) tuples
                scoring_info = []
                for word in words_counter:
                    scoring_info.append(
                        (word, words_counter[word], self.relative_word_freq[word], words_counter[word] * self.relative_word_freq[word]))

                # Sort in order of total score.
                scoring_info.sort(key=lambda x: -x[3])

                for (query_word, frequency, k_value, point_value) in scoring_info:
                    print("\"{}\" appears {} times, with k_value of {}, giving a score of {}".format(
                        query_word, frequency, k_value, point_value))
                    total_score += point_value
                print("Total score: {}".format(total_score))
                print("--------------------------------")
                sys.stdout = original_stdout
            return total_score

    doc = Document(title)

    if DEBUG_MODE:
        print("{} Book length has {} lines".format(title, len(raw_text)))

    word_list = []
    for line in raw_text:
        word_list.extend(Utils.best_tokenize(line))
    doc.num_words = len(word_list)

    doc.nltk_word_freq = nltk.FreqDist(word_list)
    doc.nltk_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(
        word_list)
    doc.nltk_trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(
        word_list)

    # Relative word frequency
    word_count = sum(doc.nltk_word_freq.values())
    doc.relative_word_freq = Counter()
    for word in doc.nltk_word_freq:
        document_word_rate = doc.nltk_word_freq[word] / word_count
        english_word_rate = word_frequency(word, "en")
        doc.relative_word_freq[word] = RARE_WORD_RELATIVE_MULTIPLIER if english_word_rate == 0 else document_word_rate / \
            english_word_rate

    return doc


# Returns a list of (Book Title, file_path) tuples
def get_book_file_list():
    filenames = [(f[:-4], join(RELATIVE_INPUT_SOURCE_DIR, f)) for f in listdir(RELATIVE_INPUT_SOURCE_DIR) if isfile(
        join(RELATIVE_INPUT_SOURCE_DIR, f))]
    return filenames


def read_book(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

        orig_lines_length = len(lines)

        # Do some minimal pre-processing to clean out non-book content.
        # Filter out the Project Gutenberg info from the top and bottom.
        proj_gut_header_end_index = 0
        proj_gut_footer_end_index = len(lines)
        for i in range(len(lines)):
            if lines[i].startswith("*** START OF THE PROJECT GUTENBERG EBOOK"):
                proj_gut_header_end_index = i
            if lines[i].startswith("*** END OF THE PROJECT GUTENBERG EBOOK"):
                proj_gut_footer_end_index = i
        del lines[0:proj_gut_header_end_index]
        del lines[proj_gut_footer_end_index:]

        num_lines_deleted = orig_lines_length - len(lines)
        assert num_lines_deleted < 400, "Sanity check - {} lines deleted".format(
            num_lines_deleted)
        return lines


def find_similar_book(query, docs_list):
    # First we do our scoring
    scores = {doc.title: doc.affinity_score(query) for doc in docs_list}

    if DEBUG_MODE:
        print(sorted(scores))

    return max(scores, key=scores.get)


def main():
    docs_list = []
    book_list = get_book_file_list()
    for title, path in book_list:
        docs_list.append(build_document(title, read_book(path)))

    # # print(docs_list[0])
    # bigrams = docs_list[0].nltk_bigram_finder
    # ngram_fd1 = bigrams.ngram_fd
    #
    # print(ngram_fd1.most_common(10))

    query = input("[Computer] What's on your mind today?\n[User] ")
    while query:
        book_suggestion_title = find_similar_book(query, docs_list)
        print("[Computer] Wow, I hadn't thought about it that way. You know, you should read {}...it might be right up your alley.\n".format(
            book_suggestion_title))
        query = input("[Computer] What are you thinking about now?\n")


main()
