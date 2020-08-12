from pdfquery import PDFQuery

import pdfminer
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator

import pandas as pd
import sys


TEXT_ELEMENTS = [
    pdfminer.layout.LTTextBox,
    pdfminer.layout.LTTextBoxHorizontal,
    pdfminer.layout.LTTextLine,
    pdfminer.layout.LTTextLineHorizontal
]


def extract_page_layouts(file):
    """
    Extracts LTPage objects from a pdf file.
    modified from: http://www.degeneratestate.org/posts/2016/Jun/15/extracting-tabular-data-from-pdfs/
    Tests show that using PDFQuery to extract the document is ~ 5 times faster than pdfminer.
    """
    laparams = LAParams()

    with open(file, mode='rb') as pdf_file:
        print("Open document %s" % pdf_file.name)
        document = PDFQuery(pdf_file).doc

        if not document.is_extractable:
            raise PDFTextExtractionNotAllowed

        rsrcmgr = PDFResourceManager()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        layouts = []
        for page in PDFPage.create_pages(document):
            interpreter.process_page(page)
            layouts.append(device.get_result())

    return layouts


def get_text_objects(page_layout):

    texts = []

    # seperate text and rectangle elements
    for elem in current_page:
        if isinstance(elem, pdfminer.layout.LTTextBoxHorizontal):
            texts.append(elem)
    return texts


def flatten(lst):
    """Flattens a list of lists"""
    return [item for sublist in lst for item in sublist]


def extract_characters(element):
    """
    Recursively extracts individual characters from
    text elements.
    """
    if isinstance(element, pdfminer.layout.LTChar):
        return [element]

    if any(isinstance(element, i) for i in TEXT_ELEMENTS):
        return flatten([extract_characters(e) for e in element])

    if isinstance(element, list):
        return flatten([extract_characters(l) for l in element])

    return []


def arrange_and_extract_text(characters, margin=0.5):
    rows = sorted(list(set(c.bbox[1] for c in characters)), reverse=True)

    row_texts = []
    for row in rows:

        sorted_row = sorted([c for c in characters if c.bbox[1] == row], key=lambda c: c.bbox[0])

        col_idx = 0
        row_text = []
        for idx, char in enumerate(sorted_row[:-1]):
            if (sorted_row[idx + 1].bbox[0] - char.bbox[2]) > margin:
                col_text = "".join([c.get_text() for c in sorted_row[col_idx:idx + 1]])
                col_idx = idx + 1
                row_text.append(col_text)
            elif idx == len(sorted_row) - 2:
                col_text = "".join([c.get_text() for c in sorted_row[col_idx:]])
                row_text.append(col_text)
        row_texts.append(row_text)
    return row_texts


def save_as_df(texts):
    columns = ['ID', 'ger_descr', 'eng_desc', 'amount']
    df = pd.DataFrame(columns=columns)
    for row in texts:
        if len(row) == 4:
            temp = pd.DataFrame([row], columns=columns)
            df = df.append(temp, ignore_index=True)
        elif len(row) == 3:
            row = row.copy()
            row.insert(0, float('nan'))
            temp = pd.DataFrame([row], columns=columns)
            df = df.append(temp, ignore_index=True)
    return df


def build_dict_hierarchy(texts):
    data = {
        'General Info': {
            'Name': texts[0][0],
            'Company Name': texts[1][0],
            'Issue': texts[2][0][:-5],
            'Year': texts[2][0][-4:]
        }
    }

    LABELS = [
        'Heading 1 is short',
        'Heading 2 which is actually very long, much longer than the first heading',
        'Heading 3 is also a little bit longer but not as long as the second'
    ]

    SUBLABELS = [
        'Segment 1-1',
        'Segment 1-2',
        'Segment 1-3',
        'Segment 2-1'
    ]

    SUMS = [
        'Something:',
        'Something else:',
        'Sum:'
    ]

    key = None
    subkey = None
    for row in texts[4:]:
        # assure that row isn't empty
        if row:
            word = row[0]
        else:
            continue

        if word in LABELS:
            key = word
            subkey = None
            data[key] = {}
        elif word in SUBLABELS:
            subkey = word
            data[key][subkey] = {}
        elif word in SUMS:
            data[key]['Sum'] = row[-1]
        elif subkey:
            ID = " ".join(row[:-1])
            data[key][subkey][ID] = row[-1]
        elif key:
            ID = " ".join(row[:-1])
            data[key][ID] = row[-1]

    return data


if __name__ == "__main__":

    data_dir = '../data/'

    if len(sys.argv) == 2:
        file = data_dir + sys.argv[1]
        print(f"using file {file}")
    else:
        file = data_dir + "example_anonymous.pdf"

    page_layouts = extract_page_layouts(file)
    print("Number of pages: %d" % len(page_layouts))

    current_page = page_layouts[0]
    text_objects =  get_text_objects(current_page)
    characters = extract_characters(text_objects)
    text = arrange_and_extract_text(characters)
    df = save_as_df(text)
    data = build_dict_hierarchy(text)

    try:
        output_file = f"{file.rsplit('.', 1)[0]}.csv"
        df.to_csv(data_dir + output_file)
        print(f"Dataframe successfully written to {output_file}")
    except:
        print(f"Dataframe could not be written to {output_file}")

    # print('\nDisplay data hierarchy:')
    # for label in data.keys():
    #     print("*" * 3, label, "*" * 3)
    #     for k, v in data[label].items():
    #         print(k)
    #         print(v, "\n")
