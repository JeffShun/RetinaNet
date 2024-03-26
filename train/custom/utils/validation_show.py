import cv2
import numpy as np
import torch


def save_result_as_image(images, scores, classes, pred_boxes, gts, lable_map, save_path):
    for i in range(images.shape[0]):
        img = images[i,0].cpu().numpy()
        img = (img * 255).astype("uint8")
        draw_img = np.zeros([img.shape[0], img.shape[1], 3])
        draw_img[:, :, 0] = img
        draw_img[:, :, 1] = img
        draw_img[:, :, 2] = img
        gti = gts.cpu().int().numpy()[i]

        probs = process_single(scores[i], classes[i], pred_boxes[i])
        for prop in probs:
            cv2.rectangle(
                draw_img,
                (prop[0], prop[1]),
                (prop[2], prop[3]),
                color=(0, 128, 256),
                thickness=2,
            )
            cv2.putText(
                draw_img,
                "%s %.3f" % (lable_map[int(prop[4])], prop[5]),
                (prop[0], prop[1] - 10),
                color=(0, 128, 256),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5,
            )

        for gt in gti:
            cv2.rectangle(
                draw_img,
                (gt[0], gt[1]),
                (gt[2], gt[3]),
                color=(0, 200, 0),
                thickness=2,
            )
            cv2.putText(
                draw_img,
                "%s" % (lable_map[int(gt[4])]),
                (gt[0], gt[1] - 10),
                color=(0, 128, 256),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.5,
            )
        cv2.imwrite(save_path+"/{}.png".format(i+1), draw_img)

def process_single(scores, labels, bboxes):
    props = []
    for box, label, score in zip(bboxes, labels, scores):
        if score < 0:
            continue
        box = box.cpu().numpy()
        cen_x, cen_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        w, h = box[2] - box[0], box[3] - box[1]
        w, h = w / 2, h / 2

        prop = [
            int(np.round(cen_x - w)),
            int(np.round(cen_y - h)),
            int(np.round(cen_x + w)),
            int(np.round(cen_y + h)),
            int(label.cpu().numpy()),
            float(score.cpu().numpy()),
            ]
        props.append(prop)
    return props



