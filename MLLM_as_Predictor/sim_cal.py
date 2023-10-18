from sentence_transformers import SentenceTransformer, util
import json
import random

ent_dict={}
ent_long_dict={}
rel_dict={}

entity_path = "./MARS/analogy_entity_to_wiki_qid.txt"
relation_path = "./MARS/analogy_relations.txt"

entity_set = []
relation_set = ["has cause", "part of", "corresponds to", "instance of", "juxtaposition to", "subject-predicate", "location", "takes place in", "opposite of", "probabilistic attribute", "has quality", "metaphor", "tool of", "antonym", "made from material", "head-modifier", "subject-object", "follow", "verb-object", "different from", "synonym", "identical to", "prerequisite", "intersection to", "has use", "contradictory to", "target of"]
answer_ent_set=[]
answer_rel_set=[]

True_answer_ent = []
True_answer_rel = []

entity_emb = []
relation_emb = []
answer_ent_emb = []
answer_rel_emb = []

model = SentenceTransformer('all-MiniLM-L6-v2')


def gen_KG():
    # create KB
    with open('./MarKG/entity2text.txt', 'r') as ent_file:
        ent_lines = ent_file.readlines()
    with open('./MarKG/entity2textlong.txt', 'r') as ent_long_file:
        ent_long_lines = ent_long_file.readlines()

    with open('./MarKG/relation2text.txt', 'r') as rel_file:
        rel_lines = rel_file.readlines()



    for line in ent_lines:
        ent1 = line.split('\t')[0]
        ent2 = line.split('\t')[1].split('\n')[0]
        ent_dict.update({ent1:ent2})

    for line in ent_long_lines:
        ent1 = line.split('\t')[0]
        ent2 = line.split('\t')[1].split('\n')[0]
        ent_long_dict.update({ent1:ent2})

    for line in rel_lines:
        rel1 = line.split('\t')[0]
        rel2 = line.split('\t')[1].split('\n')[0]
        rel_dict.update({rel1:rel2})

## generate entity set
def entity_catch():
    with open(entity_path, 'r') as ent_file:
        ent_lines = ent_file.readlines()

    for line in ent_lines:
        ent = line.split('\t')[0]
        entity_set.append(ent)



## extract answer
def answer_catch():
    answer = [json.loads(q) for q in open("./test_llava-13b-choose-answer.jsonl", "r")]
    for line in answer:
        # try:
        #     answer_total = line["text1"].split("Answer:")[1]
        #     answer_rel = answer_total.split("1.")[1].split("\n")[0]
        #     answer_ent = answer_total.split("2.")[1].split("\n")[0]
        # except:
        #     answer_rel=random.choice(relation_set)
        #     answer_ent=random.choice(entity_set)
        
        
        # if answer_rel=='':
        #     answer_rel=random.choice(relation_set)

        # if answer_ent=='':
        #     answer_ent=random.choice(entity_set)
        try:
            test = line["text"].split("Answer:")[2].split(". ")[1].split(".")[0].replace("\'",'')
            flag = 1
        except:
            flag = 0 
        try:
            answer_total = line["text"].split("Answer:")[1]
        except:
            answer_total="1. NONE. 2. NONE."

        try:
            answer_rel = answer_total.split("1. ")[1].split(".")[0].replace("\'",'').replace("\"",'')
        except:
            answer_rel='NONE'
        try:
            if flag == 1:
                answer_ent = line["text"].split("Answer:")[2].split(". ")[1].split(".")[0].replace("\'",'').replace("\"",'')
            else:
                answer_ent = answer_total.split("2. ")[1].split(".")[0].replace("\'",'').replace("\"",'')
        except:
            answer_ent='NONE'
        answer_rel_set.append(answer_rel)
        answer_ent_set.append(answer_ent)


## extract answer (irregular)
def answer_catch_com():
    answer = [json.loads(q) for q in open("./test_llava-7b-answer.jsonl", "r")]
    for line in answer:
        try:
            answer_total = line["text"].split("Answer:")[1]
        except:
            answer_total="1. NONE. 2. NONE"

        try:
            answer_rel = answer_total.split("between 'X' and 'Y'")[1].split("is")[1].split(".")[0].split("\"")[1].split("\"")[0].replace("\'",'').replace("\"",'')
        except:
            answer_rel='NONE'

        flag = 0
        try:
            answer_ent = line["text"].split('\'?\' stands for ')[1].split(".")[0].replace("\'",'').replace("\"",'')
        except:
            flag += 1       
        
        try:
            answer_ent = line["text"].split('is ')[1].split(".")[0].replace("\'",'').replace("\"",'')
        except:
            flag += 1
        
        if flag == 2: answer_ent = 'NONE' 
        if flag == 0 :  answer_ent = line["text"].split('\'?\' stands for ')[1].split(".")[0].replace("\'",'').replace("\"",'')

        answer_rel_set.append(answer_rel)
        answer_ent_set.append(answer_ent)


## answer mapping
def match():
    entity_emb = model.encode(entity_set, convert_to_tensor=True)
    relation_emb = model.encode(relation_set, convert_to_tensor=True)

    answer_ent_emb = model.encode(answer_ent_set, convert_to_tensor=True)
    answer_rel_emb = model.encode(answer_rel_set, convert_to_tensor=True)

    
    # cosine similarity
    ent_cosine_scores = util.cos_sim(answer_ent_emb, entity_emb)
    rel_cosine_scores = util.cos_sim(answer_rel_emb, relation_emb)

    ent_index = ent_cosine_scores.argmax(1)
    rel_index = rel_cosine_scores.argmax(1)


    # use miniLM, Q&A mode
    # for i in ent_index:
    #     True_answer_ent.append(entity_set[i])
    # for i in rel_index:
    #     True_answer_rel.append(relation_set[i])


    # no use nimiLM, choose mode
    for i in answer_ent_set:
        True_answer_ent.append(i)
    for i in answer_rel_set:
        True_answer_rel.append(i)

## Hit@1
def eval():
    with open('./MARS/test.json', 'r') as file:
        lines = file.readlines()
    
    num = 0
    positive_ent = 0
    positive_rel = 0

    for line in lines:
        data = json.loads(line)
        ent = data['answer']
        rel = data['relation']
        a = ent_dict[ent]
        r = rel_dict[rel]
        if a == True_answer_ent[num]:positive_ent +=1
        if r == True_answer_rel[num]:positive_rel +=1
        num += 1


    print("Ent Hit@1:%f"%(float(positive_ent)/float(num)))
    print("Rel Hit@1:%f"%(float(positive_rel)/float(num)))
    

answer_list=[]
true_list=[]
def judge_catch():
    answer = [json.loads(q) for q in open("./test_llava-7b-answer-judge.jsonl", "r")]
    for line in answer:
        total_answer = line['text']
        if ' No' in total_answer or ' invalid' in total_answer or 'not valid' in total_answer:
            answer_list.append(0)
        else:
            answer_list.append(1)

from sklearn.metrics import precision_score, recall_score, f1_score

def judge_eval():
    num=0
    for i in range(len(answer_list)):
        if i%2 ==0: 
            true_list.append(1)
        else: 
            true_list.append(0)
        if answer_list[i]==true_list[i]:
            num+=1
    p = precision_score(answer_list, true_list)
    r = recall_score(answer_list, true_list)
    f = f1_score(answer_list, true_list)
    print(num/float(len(answer_list)))
    print(p)
    print(r)
    print(f)
        


if __name__ == "__main__":
    gen_KG()
    entity_catch()
    answer_catch()
    # answer_catch_com()
    match()
    eval()
    # judge_catch()
    # judge_eval()
