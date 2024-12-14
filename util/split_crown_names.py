import random
import os

def save_list_to_txt(lst, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in lst:
            f.write(str(item) + '\n')

def split_list(input_list, split_ratio=0.8):
    random.shuffle(input_list)
    
    split_index = int(len(input_list) * split_ratio)
    
    list1 = input_list[:split_index]
    list2 = input_list[split_index:]
    
    return list1, list2

crowns_name = [f for f in os.listdir(os.path.join('/data_new2/sz_zzz/Data/Teeth/RD_1', 'crown_occ'))]
train_crowns_name, val_crowns_name = split_list(crowns_name)
train_txt = os.path.join('/data_new2/sz_zzz/Data/Teeth/RD_1', 'train.txt')
val_txt = os.path.join('/data_new2/sz_zzz/Data/Teeth/RD_1', 'val.txt')
save_list_to_txt(train_crowns_name, train_txt)
save_list_to_txt(val_crowns_name, val_txt)




