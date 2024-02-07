import spacy
import requests
import pickle
import os
from itertools import combinations
from ultralytics import YOLO
import torch
from PIL import Image
import requests
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings

warnings.filterwarnings("ignore")

nouns = ['NN', 'NNS', 'NNP', 'NNPS']

with open('./image_names.json', 'r') as f:
    image_names = json.load(f)


images_path = '../train2014/'

nlp = spacy.load('en_core_web_trf')

yolo_model = YOLO("yolov8x.pt")
yolo_model = yolo_model.to(device='cuda')

blip_checkpoint = "Salesforce/blip-image-captioning-large"
blip_processor = BlipProcessor.from_pretrained(blip_checkpoint)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_checkpoint)
blip_model.to(device='cuda')

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')



def box_generator(image):

    results = yolo_model(image)
    boxes = results[0].boxes

    image_size = results[0].orig_img.shape[0] * results[0].orig_img.shape[1]

    coords = []
    for box in boxes:

        w, h = list(map(lambda x: round(x), box.xywh[0].tolist()[2:]))
        box_size = w * h
        
        if box_size > 0.02 * image_size:
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            coords.append(cords)


    comb = list(combinations(list(range(len(coords))), 2))

    return coords, comb

def crop_generator(image, coords, comb):

    img = cv2.imread(image)

    # individual crops
    for i, coord in enumerate(coords):
        crops = img[coord[1]: coord[3], coord[0]: coord[2]]
        cv2.imwrite(os.path.join('crops', f"crops_{i+1}.jpg"), crops)
    
    
    # combined crops
    for i, c in enumerate(comb):
    
        box1 = coords[c[0]]
        box2 = coords[c[1]]

        new_x1 = min(box1[0], box2[0])
        new_y1 = min(box1[1], box2[1])
        new_x2 = max(box1[2], box2[2])
        new_y2 = max(box1[3], box2[3])

        combined_crop = img[new_y1: new_y2, new_x1: new_x2]
        cv2.imwrite(os.path.join('crops', f"combined_crops_{i+1}.jpg"), combined_crop)


def caption_generator(image, coords, comb):

    captions = []

    # complete image caption
    img = cv2.imread(image)
    unconditional_inputs = blip_processor(img, return_tensors="pt")
    unconditional_inputs.to(device='cuda')
    unconditional_output = blip_model.generate(**unconditional_inputs)
    unconditional_text = blip_processor.decode(unconditional_output[0], skip_special_tokens=True)
    captions.append(unconditional_text)

    # individual crops captions
    for i in range(len(coords)):
        individual_crop_image = os.path.join('crops', f'crops_{i+1}.jpg')
        img = cv2.imread(individual_crop_image)

        unconditional_inputs = blip_processor(img, return_tensors="pt")
        unconditional_inputs.to(device='cuda')
        unconditional_output = blip_model.generate(**unconditional_inputs)
        unconditional_text = blip_processor.decode(unconditional_output[0], skip_special_tokens=True)
        captions.append(unconditional_text)

    # combined crops captions
    for i in range(len(comb)):
        if i <= 10:
            combined_crop_image = os.path.join('crops', f'combined_crops_{i+1}.jpg')
            img = cv2.imread(combined_crop_image)

            unconditional_inputs = blip_processor(img, return_tensors="pt")
            unconditional_inputs.to(device='cuda')
            unconditional_output = blip_model.generate(**unconditional_inputs)
            unconditional_text = blip_processor.decode(unconditional_output[0], skip_special_tokens=True)
            captions.append(unconditional_text)

    return captions



def question_generator(captions):

    ques_ans_dict = {}
    
    def get_question(sentence, answer, model, tokenizer):
        max_len = 256
        text = "context: {} answer: {}".format(sentence, answer)
        encoding = tokenizer.encode_plus(text, max_length=max_len, pad_to_max_length=False, truncation=True, return_tensors="pt")
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
        outs = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=300)


        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        question = dec[0].replace("question:", "")
        question = question.strip()
        return question


    for cap in captions:
    
        answers = []
        sent = nlp(cap)
        for token in sent:
            if token.tag_ in nouns:
                answers.append(str(token))

        for ans in answers:
            ques = get_question(cap, ans, question_model, question_tokenizer)
            if ans not in ques:
                ques_ans_dict[ques] = ques_ans_dict.get(ques, set()) | set((ans,))
            #print(f"question: {ques}, answer: {ans}")

    return ques_ans_dict




if __name__ == "__main__":

    for i in range(0, len(image_names) + 1):

        print(i)

        image_name = image_names[i]
        image = images_path + image_name

        coords, comb = box_generator(image)
        print(f"number of boxes: {len(coords)}, number of combinations:{len(comb)}")

        crop_generator(image, coords, comb)
        print("Crops Generated")

        captions = caption_generator(image, coords, comb)
        print("Captions Generated")

        ques_ans_dict = question_generator(captions)
        print("Questions Generated")

        file_name = image_name.split('.')[0]

        with open(f"./captions/{file_name}.json", 'w') as f:
            json.dump(captions, f)

        with open(f"./questions/{file_name}.pickle", 'wb') as f:
            pickle.dump(ques_ans_dict, f)

        for root, dirs, files in os.walk('./crops/'):
            for f in files:
                os.remove(os.path.join(root, f))

        print()

