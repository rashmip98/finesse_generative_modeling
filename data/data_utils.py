import json
import gensim
import numpy as np

# parse through product_data.json and extract product_id to prod_title description
def create_prod_title_json():
    with open('data/product_data.json','r') as j_file:
        j_object = json.load(j_file)

    new_j_obj = {}
    vocab = set()

    for k, v in j_object.items():
        # if v['main_category'] == "Sets":
        #     continue
        title = v['title'].lower().split()[1:]
        new_j_obj[k] = title

        vocab.update(title)

    with open("data/prod_title_map.json", "w") as outfile:
        json.dump(new_j_obj, outfile)
    
    with open("vocab.txt", "w") as f:
        for item in vocab:
            f.write(item + "\n")

def create_sentence_embeddings_and_save():
    with open('data/prod_title_map.json','r') as j_file:
        j_object = json.load(j_file)
    sentences = []
    for k, v in j_object.items():
        sentences.append(v)
    model = gensim.models.Word2Vec(sentences=sentences, vector_size=300, min_count=1)

    sentence_embedding_dict = {}
    for k,v in j_object.items():
        vec = np.zeros(300)
        for word in v:
            vec += model.wv[word]
        sentence_embedding_dict[k] = vec/len(v)
    
    np.savez('data/prod_embeddings.npz', **sentence_embedding_dict)


def main():
    # create_prod_title_json()
    create_sentence_embeddings_and_save()

if __name__ == "__main__":
    main()