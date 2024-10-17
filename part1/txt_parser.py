import chardet


def __detect_encoding(file_path: str):
    with open(file_path, 'rb') as file:
        detector = chardet.universaldetector.UniversalDetector()
        for line in file:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    return detector.result['encoding']


def read_file_with_encoding(file_path):
    with open(file_path, 'r', encoding=__detect_encoding(file_path)) as file:
        return file.read()
