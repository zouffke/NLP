import spacy
import pandas as pd

from global_scripts.txt_parser import read_file_with_encoding
from spacy.matcher import Matcher


class Analyser:
    def __init__(self,
                 input_data: str,
                 is_file: bool = True,
                 nlp_type: str = "Model",
                 nlp_model: str = "en_core_web_sm"):
        if nlp_type == "Model":
            self.__nlp = spacy.load(nlp_model)
        elif nlp_type == "Blank":
            self.__nlp = spacy.blank(nlp_model)
        else:
            raise Exception("Invalid type for nlp_type. Only 'Model' and 'Blank' are accepted")

        if is_file:
            text = read_file_with_encoding(input_data)
        else:
            text = input_data

        self.__doc = self.__nlp(text)
        self.__matcher = Matcher(self.__nlp.vocab)
        super().__init__()

    def analyse_sentences(self, do_print=False):
        if not do_print:
            df = pd.DataFrame(columns=['SENTENCE', 'WORD', 'POS', 'DEP', 'DEP_ON', 'POS_EXPLAIN', 'DEP_EXPLAIN'])

            for sents in self.__doc.sents:
                df.loc[len(df)] = {'SENTENCE': sents.text}
                for ent in sents:
                    df.loc[len(df)] = {'WORD': ent.text,
                                       'POS': ent.pos_,
                                       'DEP': ent.dep_,
                                       'DEP_ON': ent.head.text,
                                       'POS_EXPLAIN': spacy.explain(ent.pos_),
                                       'DEP_EXPLAIN': spacy.explain(ent.dep_)
                                       }

            return df
        else:
            for sents in self.__doc.sents:
                print(f'sentence:\n\t{sents}\n{"\t" * 2}Sentence Parts:')
                for ent in sents:
                    print(
                        f'{"\t" * 3}{ent.text} {ent.pos_}({spacy.explain(ent.pos_)}) {ent.dep_}({spacy.explain(ent.dep_)}) {ent.head.text}')
                print(f'\n{"=" * 20}\n')

    def named_entities(self):
        df = pd.DataFrame(columns=['TEXT', 'LABEL'])
        for ent in self.__doc.ents:
            df.loc[len(df)] = {'TEXT': ent.text, 'LABEL': ent.label_}

        return df

    def find_matches(self, pattern: [], analyser_name="analyser"):
        df = pd.DataFrame(columns=['MATCHES'])
        try:
            self.__matcher.remove(analyser_name)
        except ValueError:
            pass
        self.__matcher.add(analyser_name, [pattern])
        matches = self.__matcher(self.__doc)
        for match_id, start, end in matches:
            df.loc[len(df)] = {'MATCHES': self.__doc[start:end].text}

        return df
