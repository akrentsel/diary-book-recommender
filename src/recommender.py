
import collections
import re
from enum import Enum
from os import listdir
from os.path import isfile, join

from wordfreq import word_frequency

# Consts and enums
RELATIVE_INPUT_SOURCE_DIR = "books/"
DEBUG_MODE = False
# This is the ratio to use if a legitimate english word appears in the document that appears ~0 times in English normally.
RARE_WORD_RELATIVE_MULTIPLIER = 1000


class Capabilities(Enum):
    # Simple number of times that the word appears in the document source.
    SIMPLE_WORD_FREQ = 1
    # The ratio between the appearance rate in the doc, vs. in all English text.
    RELATIVE_WORD_FREQ = 2


def naive_tokenize(string):
    """
    Naive implementation of getting tokens out of a string.
    """
    return re.findall(r'\w+', string)


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
                # zipf frequencies are effectively on a scale of 0 to 8. They correspond to
                # the base-10 logarithm of the number of times it appears per billion words.
                # See https://pypi.org/project/wordfreq/ for details.
                words = naive_tokenize(raw_query)
                words = filter(lambda word: zipf_frequency(
                    word, "en") < 6, words)
            words_counter = Counter(process_query(raw_query))

            # Now, we will calculate a score using our relative word freq.
            return sum([self.relative_word_freq[word] * words_counter[word] for word in words_counter])

    doc = Document(title)

    if DEBUG_MODE:
        print("{} Book length is {}".format(title, len(raw_text)))

    # simple frequency map
    doc.capabilities_list.append(Capabilities.SIMPLE_WORD_FREQ)
    word_list = re.findall(r'\w+', " ".join(raw_text).lower())

    def spammy_word_filter(word):
        """
        Returns true if a word is okay, false if a word is spammy.
        """
        spam_checks = [word.startswith("_"), word.endswith("_")]
        return not any(spam_checks)
    doc.simple_word_freq = collections.Counter(
        filter(spammy_word_filter, word_list))

    # Relative word frequency
    if Capabilities.SIMPLE_WORD_FREQ in doc.capabilities_list:
        doc.capabilities_list.append(Capabilities.RELATIVE_WORD_FREQ)
        word_count = sum(doc.simple_word_freq.values())
        doc.relative_word_freq = collections.Counter()
        for word in doc.simple_word_freq:
            document_word_rate = doc.simple_word_freq[word] / word_count
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
        # 1. Filter out the Project Gutenberg info from the top and bottom.
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
    scores = [{doc.title, doc.affinity_score(query)} for doc in docs_list]

    if DEBUG_MODE:
        print(sorted(scores))

    return max(scores, key=scores.get)


def main():
    docs_list = []
    book_list = get_book_file_list()
    for title, path in book_list:
        docs_list.append(build_document(title, read_book(path)))


main()
