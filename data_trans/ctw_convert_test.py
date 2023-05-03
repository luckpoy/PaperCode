# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: synth_convert.py

import os
import re
import jsonlines
import numpy as np
import sys
sys.path.append("/home/tml/bo/CRAFT/CRAFT-Remade/train_ctw")

def get_word_polygon(char_dicts):
        ############################
        # This function is extracting words' polygon from some char's polygons
        ############################
        up_list = list() 
        bot_list = list() 
        lef_list = list() 
        rig_list = list()
        for i in range(len(char_dicts)):
            up_list.append(char_dicts[i]["adjusted_bbox"][1]) 
            bot_list.append(char_dicts[i]["adjusted_bbox"][1]+char_dicts[i]["adjusted_bbox"][3])  
            lef_list.append(char_dicts[i]["adjusted_bbox"][0])  
            rig_list.append(char_dicts[i]["adjusted_bbox"][0]+char_dicts[i]["adjusted_bbox"][2]) 
        up = min(up_list)
        bot = max(bot_list)
        lef = min(lef_list)
        rig = max(rig_list)
        return up, bot, lef, rig

class CTWConvertor:
    def __init__(self, jsonl_path, image_root):
        super(CTWConvertor, self).__init__()
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.image_name_list, self.word_boxes_list, self.char_boxes_list, self.texts_list = self.__load_jsonl()

    def __load_jsonl(self):
        ###################################
        # word_boxes_list format:
        # [[[],[],[],[]],  -----word box
        #  [           ], [],...]  
        # char_boxes_list format:
        # [[[],[],[],[]],[   ],  --- char boxes for one word
        #  []...]
        # texts_list format:
        # [aa,aaa,aaaaa,aaaaaa,...]
        ###################################
        image_name_list = list()
        word_boxes_list = list()
        char_boxes_list = list()
        texts_list = list()
        with open(self.jsonl_path, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                image_name_list.append(item["file_name"])
                word_box_list = list()
                chars_box_list = list()
                # annotations
                for i in range(len(item["proposals"])):
                    text = ''
                    char_box_list = list()
                    char_box_list.append(item["proposals"][i]["polygon"])
                    # up, bot, lef, rig = get_word_polygon(item["proposals"][i])
                    # word_box_list.append([[lef,up], [rig,up], [rig,bot], [lef,bot]])
                    chars_box_list.append(np.array(char_box_list))
                # Hard samples 
                try:   
                    for k in range(len(item["ignore"])):
                        text = ''
                        char_box_list = list()
                        for l in range(len(item["ignore"][k])):
                            char_box_list.append(item["ignore"][k][l]["polygon"])
                            text += 'a'
                        texts_list.append(text)
                        up, bot, lef, rig = get_word_polygon(item["ignore"][k])
                        word_box_list.append([[lef,up], [rig,up], [rig,bot], [lef,bot]])
                        chars_box_list.append(np.array(char_box_list))
                except:
                    pass
                # word_boxes_list.append(np.array(word_box_list))
                char_boxes_list.append(chars_box_list)
        return image_name_list, word_boxes_list, char_boxes_list, texts_list
    

    @staticmethod
    def split_text(texts):
        split_texts = list()
        for text in texts:
            text = re.sub(' ', '', text)
            split_texts += text.split()
        return split_texts

    @staticmethod
    def swap_box_axes(boxes):
        if len(boxes.shape) == 2 and boxes.shape[0] == 2 and boxes.shape[1] == 4:
            # (2, 4) -> (1, 4, 2)
            boxes = np.array([np.swapaxes(boxes, axis1=0, axis2=1)])
        else:
            # (2, 4, n) -> (n, 4, 2)
            boxes = np.swapaxes(boxes, axis1=0, axis2=2)
        return boxes

    def convert_to_craft(self):
        sample_list = list()
        for image_name, char_boxes in zip(self.image_name_list,self.char_boxes_list):
            image_path = os.path.join(self.image_root, image_name)
            sample_list.append([image_path, char_boxes])
        return sample_list

    def convert_to_test(self):
        sample_list = list()
        for image_name, char_boxes in zip(self.image_name_list,self.char_boxes_list):
            sample_list.append([image_name, char_boxes])
        return sample_list

def polygon_area(points):
    """返回多边形面积
    """
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return abs(area / 2)


if __name__ == '__main__':
    jsonl_path=r'/home/tml/boo/tml/bo/CRAFT/CRAFT-Remade/ctw/modify/test_cls.jsonl'
    image_root=r'/home/tml/bo/CRAFT/CRAFT-Remade/ctw/images-test'
    CTW_text_convertor = CTWConvertor(jsonl_path,image_root)
    craft_sample_list = CTW_text_convertor.convert_to_test()

    for line in craft_sample_list:
        img_path = os.path.join(image_root, line[0])

        char_box = line[1].copy()[0]
        for i in range(1, len(line[1])):
            char_box = np.concatenate((char_box, line[1].copy()[i]), axis=0)
        sum_area = 0
        for box in char_box:
            sum_area = sum_area + polygon_area(box)
        average = sum_area / len(char_box)
        if average>500:
            continue
        else:
            print(line[0])
            a = line[0]
            cmd = "sed -i '/" + a  + "/d'  /home/tml/boo/tml/bo/CRAFT/CRAFT-Remade/ctw/modify/test_cls.jsonl"
            os.system(cmd)