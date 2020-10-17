from scipy.misc import imresize
import numpy as np


##################################################################################
############################Do not modify opts anymore.###########################
######################Becuase of synchronization of options#######################
##################################################################################

def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def crop_image(img, bbox, img_size=[107,107], padding=16, valid=False):
    ## img_size = [w,h]
    x,y,w,h = np.array(bbox,dtype='float32')

    half_w, half_h = w/2, h/2
    center_x, center_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding * w/img_size[0]
        pad_h = padding * h/img_size[1]
        half_w += pad_w
        half_h += pad_h

    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

    if min_x >=0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y-min_y, max_x-min_x, 3), dtype='uint8')
        cropped[min_y_val-min_y:max_y_val-min_y, min_x_val-min_x:max_x_val-min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    scaled = imresize(cropped, (img_size[1],img_size[0]))
    return scaled

def samples2maskroi(samples,receptive_field, cshape,padded_scene_size,padding_ratio):
    # rois is from domain of original image axis
    # receptive field can be subtracted to x2,y2

    # ratios between original image and resized_image
    cur_resize_ratio = cshape / padded_scene_size
    rois = np.copy(samples)

    # xywh -> x1y1x2y2
    rois[:, 2:4] += rois[:, 0:2]

    # padding application
    rois_paddings = (rois[:,2:4]-rois[:,0:2])*(padding_ratio-1.)/2.
    rois[:,0:2]-=rois_paddings
    rois[:,2:4]+=rois_paddings


    rois[:, 0] *= cur_resize_ratio[0]
    rois[:, 1] *= cur_resize_ratio[1]
    rois[:, 2] = np.maximum(rois[:,0]+1,rois[:, 2]*cur_resize_ratio[0] - receptive_field)
    rois[:, 3] = np.maximum(rois[:,1]+1,rois[:, 3]*cur_resize_ratio[1] - receptive_field)


    return rois
