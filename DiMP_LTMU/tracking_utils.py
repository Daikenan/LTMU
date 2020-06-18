import cv2
import numpy as np


def show_res(im, box, win_name,update=None,score=None,frame_id=None,mask=None,score_max=None, groundtruth=None, can_bboxes=None):
    cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)
    cv2.rectangle(im, (box[1], box[0]),
                  (box[3], box[2]), [0, 255, 255], 2)

    if mask is not None:
        im[:, :, 1] = (mask > 0) * 255 + (mask == 0) * im[:, :, 1]
    if can_bboxes is not None:
        can_bboxes = np.array(can_bboxes, dtype=np.int32)
        for i in range(len(can_bboxes)):
            cv2.rectangle(im, (can_bboxes[i, 0], can_bboxes[i, 1]),
                          (can_bboxes[i, 0] + can_bboxes[i, 2], can_bboxes[i, 1] + can_bboxes[i, 3]), [255, 0, 0], 2)
    if groundtruth is not None and not groundtruth[frame_id][0] == np.nan:
        groundtruth = groundtruth.astype("int16")
        cv2.rectangle(im, (groundtruth[frame_id][0], groundtruth[frame_id][1]),
                      (groundtruth[frame_id][0] + groundtruth[frame_id][2],
                       groundtruth[frame_id][1] + groundtruth[frame_id][3]), [0, 0, 255], 2)
    if update is not None:
        cv2.putText(im, str(update), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    if score_max is not None:
        cv2.putText(im, str(score_max), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    if score is not None:
        cv2.putText(im, str(score), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    if frame_id is not None:
        cv2.putText(im, str(frame_id), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.imshow(win_name, im)
    cv2.waitKey(1)


def compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou


def process_regions(regions):
    # regions = np.squeeze(regions, axis=0)
    regions = regions / 255.0
    regions[:, :, :, 0] = (regions[:, :, :, 0] - 0.485) / 0.229
    regions[:, :, :, 1] = (regions[:, :, :, 1] - 0.456) / 0.224
    regions[:, :, :, 2] = (regions[:, :, :, 2] - 0.406) / 0.225
    regions = np.transpose(regions, (0, 3, 1, 2))
    # regions = np.expand_dims(regions, axis=0)
    # regions = np.tile(regions, (2,1,1,1))

    return regions


class Region:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

