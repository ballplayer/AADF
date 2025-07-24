import numpy as np
import cv2
import torch
from torchvision.utils import save_image
import os


def save_adv(images, attack_success, adv, dir_epsilon, attack_success_nums, file_content):
    cnt = 0
    print("imgs have been saved in {:}".format(dir_epsilon))
    file_content.write("imgs have been saved in {:}\n".format(dir_epsilon))

    record_site_text = 'record_site.txt'
    file = open(record_site_text, mode='w')
    file.write(dir_epsilon)
    file.close()

    for i in range(len(images)):
        if attack_success[i]:
            img_path = dir_epsilon + '/' + str(attack_success_nums + cnt) + '.png'
            adv_img_path = dir_epsilon + '/' + str(attack_success_nums + cnt) + '_adv.png'
            adv_img = (adv[i] if not isinstance(adv[i], np.ndarray) else torch.from_numpy(adv[i]))

            save_image(images[i], img_path)
            save_image(adv_img, adv_img_path)
            cnt += 1


def od_save_adv(img, pred, attack_success, adv_image, adv_pred, save_path, save_boxes_path):
    save_image_without_boxes(img=img.transpose(1, 2, 0).copy(), title="{:}".format(attack_success), save_path=save_path)
    save_image_with_boxes(img=img.transpose(1, 2, 0).copy(), boxes=pred[1], pred_cls=pred[0],
                          title="{:}".format(attack_success), save_boxes_path=save_boxes_path)

    save_image_without_boxes(img=adv_image.transpose(1, 2, 0).copy(), title="{:}_adv".format(attack_success), save_path=save_path)
    save_image_with_boxes(img=adv_image.transpose(1, 2, 0).copy(), boxes=adv_pred[1], pred_cls=adv_pred[0],
                          title="{:}_adv".format(attack_success), save_boxes_path=save_boxes_path)


def save_image_without_boxes(img, title, save_path):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    path = os.path.join(save_path, title + '.jpg')
    cv2.imwrite(path, img)


def save_image_with_boxes(img, boxes, pred_cls, title, save_boxes_path):
    text_size = 1
    text_th = 3
    rect_th = 2
    for i in range(len(boxes)):
        cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1])),
                      color=(0, 255, 0), thickness=rect_th)
        # Write the prediction class
        cv2.putText(img, pred_cls[i], (int(boxes[i][0][0]), int(boxes[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    (0, 255, 0), thickness=text_th)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    path = os.path.join(save_boxes_path, title + '.jpg')
    cv2.imwrite(path, img)
