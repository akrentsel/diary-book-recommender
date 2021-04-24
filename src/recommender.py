
from enum import Enum
from os import listdir
from os.path import isfile, join

RELATIVE_INPUT_SOURCE_DIR = "books/"
DEBUG_MODE = True


class Capabilities(Enum):
    SIMPLE_WORD_FREQ = 1


def build_document(title, raw_text):
    # Hide the document class so it has to be instantiated via this factory.
    class Document:
        def __init__(self, title):
            self.title = title
            self.capabilities_list = []

        def __repr__(self):
            return "\"{}\"".format(self.title)
    print("Book length is {}".format(len(raw_text)))

    doc = Document(title)
    return doc


# Returns a list of (Book Title, file_path) tuples
def get_book_file_list():
    filenames = [(f[:-4], join(RELATIVE_INPUT_SOURCE_DIR, f)) for f in listdir(RELATIVE_INPUT_SOURCE_DIR) if isfile(
        join(RELATIVE_INPUT_SOURCE_DIR, f))]
    return filenames


def read_book(file_path):
    with open(file_path, "r") as file:
        return file.readlines()


def main():
    docs_list = []
    book_list = get_book_file_list()
    for title, path in book_list:
        docs_list.append(build_document(title, read_book(path)))
        break
    print(docs_list)


main()
