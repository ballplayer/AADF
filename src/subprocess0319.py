import json
import torch
import common

if __name__ == '__main__':
    with open("AE_thread_info.json", 'r') as load_f:
        load_dict = json.load(load_f)
    info1 = load_dict["model_name"]
    info2 = load_dict["dataset_name"]
    info3 = load_dict["task"]
    info4 = load_dict["save_path"]
    info5 = load_dict["save_boxes_path"]
    # info6 = load_dict["substitute"]
    info7 = load_dict["method_name"]
    # info8 = load_dict["targeted"]
    # info9 = load_dict["distance_str"]
    info10 = load_dict["epsilons"]

    common.attacks(model_name=info1, dataset_name=info2, task=info3, save_path=info4,
                   save_boxes_path=info5, 
                   method_name=info7,
                   epsilons=info10)

