import datetime
def gettimestamp():
    now = datetime.datetime.now()
    nowstr = now.strftime("%Y%m%d_%H%M%S")
    return nowstr




import json
import os
def convert_inter_to_saveable(index):
    json_index = []
    for i in index:
        json_index.append(str(i))
    return json_index

def convert_inter_to_loaded(json_index):
    index = []
    for str_i in json_index:
        index.append(int(str_i))
    return tuple(index)

def save_interactions_json(indices, file_path):
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    json_indices = [convert_inter_to_saveable(index) for index in indices]
    # with open(file_path, 'w', encoding='utf-8') as f:
    #     json.dump(json_indices, f, ensure_ascii=False, indent=4)
    # https://www.onlinegdb.com/Gqa_f9hys
    # json_str = re.sub(r"(?<=\[)[^\[\]]+(?=])", repl_func, json_str)

    indent = 4
    json_str = ""
    json_str += "[\n"
    for jj,index in enumerate(json_indices):
        line = ""
        line += " "*indent+"["
        for ii,str_i in enumerate(index):
            line+="\""+str_i+"\""
            if ii!=len(index)-1:
                line+=", "
        line += "]"
        if jj!=len(json_indices)-1:
            line+=","
        json_str += line + "\n"
    json_str += "]"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json_str)

def load_interactions_json(file_path):
    with open(file_path) as f:
        json_indices = json.load(f)
    indices = [convert_inter_to_loaded(json_index) for json_index in json_indices]
    return indices







import datetime
def gettimestamp():
    now = datetime.datetime.now()
    nowstr = now.strftime("%Y%m%d_%H%M%S")
    return nowstr