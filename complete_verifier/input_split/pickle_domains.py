import pickle
import os
from pathlib import Path

def pickle_domains(lb, x_L, x_U, rep, exp, lb_2):
    base_path = os.path.realpath(__file__).replace("pickle_domains.py", "") + "/pickled/"
    dir = base_path + exp
    Path(dir).mkdir(parents=True, exist_ok=True)
    _, _, files = next(os.walk(dir))
    file_count = len(files)
    
    dict_ = {
        "lb": lb,
        "x_L": x_L,
        "x_U": x_U,
        "rep": rep,
        "lb_plain": lb_2
    }

    with open(f'{dir}/{file_count}.pickle', 'wb') as handle:
        pickle.dump(dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return