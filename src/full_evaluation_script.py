import sys
sys.path.append("src/")
import json
from langchain_community.vectorstores import FAISS
from datasets import load_dataset
from mamba_ralm import MambaRALM
from dolly_ralm import DollyRALM
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from triviaqa_evaluation import evaluate_triviaqa
from tqdm import tqdm
from datetime import datetime
import vector_store
import json

def instantiate_vector_store():

    triviaqa_vector_store = vector_store.RAGVectorStore()

    return triviaqa_vector_store.load_db("triviaqa_vector_store")

def get_evaluation_ds():

    evaluation_dataset_name = "wikipedia-dev-sample"
    evaluation_ds_filepath = "{ds_name}.json".format(ds_name=evaluation_dataset_name)

    with open (evaluation_ds_filepath, "r") as json_file:
        eval_ds = json.load(json_file)

    return eval_ds

def generate_and_evaluate_model(model_name=None, k=None, vector_db=None, eval_ds=None):

    if model_name == "mamba":

        model = MambaRALM("havenhq/mamba-chat", vector_db)

    elif model_name == "dolly":

        model = DollyRALM(vector_db=vector_db)

    else:

        raise RuntimeError("Model Not Defined")

    generated_responses = {}

    K_VALUE = k

    for question in tqdm(eval_ds):

        question_text = question["Question"]

        model_prediction = model.predict(question_text, k=K_VALUE)

        if type(model_prediction) == list:
            model_prediction = model_prediction[0]

        generated_responses[question["QuestionId"]] = model_prediction

    wiki_dev_text_keyed = {question["QuestionId"]: question["Answer"] for question in eval_ds}

    num_total = len(generated_responses)
    num_correct = 0

    for question_key in generated_responses:

        correct = False

        for alias in wiki_dev_text_keyed[question_key]["NormalizedAliases"]:

            if alias in generated_responses[question_key].lower():
                correct = True
                break

        if correct:
            num_correct += 1

    accuracy = num_correct / num_total

    print(accuracy)

    with open("{mn}_k{kv}.txt".format(mn=model_name, kv=k), "w") as file:

        file.write(str(accuracy))

if __name__ == "__main__":

    db = instantiate_vector_store()

    eval_ds = get_evaluation_ds()

    models = ["mamba", "dolly"]

    for modeln in models:

        for k in range(6, 15):

            try:
            
                generate_and_evaluate_model(model_name=modeln, k=k, vector_db=db, eval_ds=eval_ds)
            
            except:

                print("model generation failed on {modeln}, k={l}".format(modeln=modeln, k=k))