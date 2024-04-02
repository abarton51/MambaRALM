import shutil
import os
from tqdm import tqdm

def get_filepath(entry : dict):
    return [page['Filename'] for page in entry["EntityPages"]]

invalid_chars = {
    ":": "_",
    '\"': "_",
    "*": "_",
    "?": "_",
}

def replace_invalid_characters(filename : str):

    translated_filename = filename.translate(invalid_chars)

    if translated_filename == 'John_"Hannibal"_Smith.txt':

        translated_filename = 'John__Hannibal__Smith.txt'

    if translated_filename == 'Toys_"R"_Us.txt':

        translated_filename = 'Toys__R__Us.txt'

    return translated_filename

def isolate_relevant_wikipedia_evidence():

    new_folder = 'data/triviaqa-rc/wikipedia_relevant_evidence/'
    existing_folder = "data/triviaqa-rc/evidence/wikipedia/"

    with tqdm(total=len(eval_ds), desc="Moving Files") as pbar:

        for qa in eval_ds:
            filepaths = get_filepath(qa)
            
            for filepath in filepaths:

                filename = replace_invalid_characters(filepath.split("\\")[-1].split("/")[-1])
                shutil.copy(os.path.join(existing_folder, filename), new_folder)
            
            pbar.update(1)