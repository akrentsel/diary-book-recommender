
def build_document(title, raw_text):
    # Hide the document class so it has to be instantiated via this factory.

    class Document:
        def __init__(self, title):
            self.title = title


def main():
    print("Hello World")


main()
