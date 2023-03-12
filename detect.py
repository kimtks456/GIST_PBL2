import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import bunny_preprocessing
from backHead_classification import HeadClassification
import face_recog_image
import iou
from backHead_classification.backHead_classification_CNN import HeadClassificationCNN


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    ########################################
    ################ San ###################

    # load replacement image
    pre_img = "peach.png"
    bunny_preprocessing.bunny_preprocessing(pre_img)
    replacedImg = cv2.imread("pre_" + pre_img)

    # # load back head classifier
    # # knn = HeadClassification.main() # using KNN
    # headmodel = HeadClassificationCNN.CustomConvNet(2) # using cumstomCNN
    # headmodel.load_state_dict(torch.load('custom_model.pth'))
    # headmodel.eval()

    # load heroine face recognition model
    facemodel = face_recog_image.FaceRecog()

    # # Load heroines coordinates
    # f = open("mudo_center.txt", 'r')
    # lines = f.readlines()
    # frame = 0

    ################ San ###################
    ########################################

    heroCoord = [[0, 0, 0, 0] for _ in range(5)]

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        ########################################
        ################ San ###################

        # # Heroine coordinate in this frame
        # line = lines[frame].strip().split(', ')
        # frame += 1
        # heroine_coord = None
        # if line[1] != 'None':
        #     heroine_coord = torch.from_numpy(np.array(line[1:]).astype(np.float32))
        #     margin = 100
        #     margin = torch.tensor(np.array(margin).astype(np.float32))
        #     x_lower = heroine_coord[1] - margin
        #     x_upper = heroine_coord[1] + margin
        #     y_lower = heroine_coord[0] - margin
        #     y_upper = heroine_coord[0] + margin
        # print("\nTEST###################")
        # print(line)

        # save heroine coordinate

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # print('This is im0 size', len(im0[0]), len(im0[1]), len(im0[2]))
            # cv2.imshow('im0', im0)
            # cv2.waitKey(0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Get max_bbox
                count, max_vol, max_idx = 0, 0.0, 0
                max_box = np.zeros((1, 4), dtype=float)
                for *xyxy, conf, cls in reversed(det):
                    res, coor = [], []
                    # xyxy to numpy
                    for i in xyxy:
                        coor.append(i.numpy())
                    res.append(coor)
                    res = np.array(res, dtype=float)
                    vol = (res[0, 2] - res[0, 0]) * (res[0, 3] - res[0, 1])
                    label = f'{names[int(cls)]} {conf:.2f}'
                    if opt.heads or opt.person:
                        if 'head' in label and opt.heads:
                            if vol > max_vol and opt.heads:
                                max_vol = vol
                                max_idx = count
                                max_box = res
                    count += 1

                # Write results
                count = 0
                for *xyxy, conf, cls in reversed(det):
                    res, coor = [], []
                    # xyxy to numpy
                    for i in xyxy:
                        coor.append(i.numpy())
                    res.append(coor)
                    res = np.array(res, dtype=int)

                    # Bin // remove background in bunny.png
                    img_r = cv2.resize(replacedImg, (res[0, 2] - res[0, 0], res[0, 3] - res[0, 1]))
                    img_sum = img_r[:, :, 0] + img_r[:, :, 1] + img_r[:, :, 2]
                    mask = np.where(img_sum > 0, 1, 0)
                    (w, h,) = img_sum.shape
                    mask_rgb = np.zeros((w, h, 3), dtype=int)
                    mask_rgb[:, :, 0] = mask
                    mask_rgb[:, :, 1] = mask
                    mask_rgb[:, :, 2] = mask

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        if opt.heads or opt.person:
                            H, W, c = im0.shape
                            im0_crop = im0[res[0, 1]:res[0, 3], res[0, 0]:res[0, 2], :]
                            location = [res[0, 1], res[0, 3], res[0, 0], res[0, 2]]
                            # print(location, "################")
                            temp_x1 = res[0, 1] - 70
                            temp_x2 = res[0, 3] + 70
                            temp_y1 = res[0, 0] - 70
                            temp_y2 = res[0, 2] + 70

                            if temp_x1 < 0:
                                temp_x1 = 1
                            elif temp_x2 > H:
                                temp_x2 = H - 1
                            elif temp_y1 < 0:
                                temp_y1 = 1
                            elif temp_y2 > W:
                                temp_y2 = W - 1
                            im0_crop_rr = im0[temp_x1:temp_x2, temp_y1:temp_y2, :]
                            ########################################
                            ################ San ###################
                            # if 'head' in label and opt.heads:
                            #     # middle of box coordinates heroine
                            #     mid_coor = np.array([int((res[0, 0] + res[0, 2]) / 2), int((res[0, 1] + res[0, 3]) / 2)])
                            #     mid_coor = torch.from_numpy(mid_coor)
                            #     if heroine_coord is not None and x_lower <= mid_coor[0] <= x_upper and \
                            #             y_lower <= mid_coor[1] <= y_upper:
                            #         # print("################# Jaesuk detected ########################################")
                            #         continue

                            # if 'head' in label and opt.heads and count != max_idx:
                            # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                            # # KNN : classify back heads
                            # im0_crop_temp = HeadClassification.resize_image(im0_crop)
                            # im0_crop_flatten = im0_crop_temp.flatten()
                            # predict_label = knn.predict([im0_crop_flatten])
                            # if predict_label == 0: # im0_crop is face
                            #     im0[res[0, 1]:res[0, 3], res[0, 0]:res[0, 2], :] = np.where(mask_rgb > 0, img_r,
                            #                                                                 im0_crop)

                            # # CNN : classify back heads = label 0
                            # if 'head' in label and opt.heads:
                            #     predict_label = HeadClassificationCNN.predict_image(im0_crop, headmodel)
                            #     if predict_label == 1: # im0_crop is not heroine
                            #         im0[res[0, 1]:res[0, 3], res[0, 0]:res[0, 2], :] = np.where(mask_rgb > 0, img_r,
                            #                                                                     im0_crop)

                            # if 'head' in label and opt.heads:
                            #     cv2.imshow('yolo', cv2.cvtColor(im0_crop, cv2.COLOR_BGR2RGB))
                            #     cv2.waitKey(0)

                            # Classify Heroine or Not
                            if 'head' in label and opt.heads:
                                temp_iou = False
                                for i in range(len(heroCoord)):
                                    if len(heroCoord[i]) > 0:
                                        if iou.IoU(res[0], heroCoord[i]) > 0.9:
                                            temp_iou = True
                                            break
                                # print(im0_crop_rr, im0_crop_rr.shape, res)
                                if facemodel.get_frame(cv2.cvtColor(im0_crop_rr, cv2.COLOR_BGR2RGB), location,
                                                       temp_iou):  # not heroine
                                    # if 'head' in label and opt.heads and facemodel.get_frame(im0_crop) is not True: # heroine
                                    # im0[res[0,1]:res[0,3], res[0,0]:res[0,2],:] = np.where(mask_rgb > 0, img_r, im0_crop)
                                    # im0[res[0,1]:res[0,3], res[0,0]:res[0,2],:] = np.where(mask_rgb == 253, img_r, im0_crop)
                                    heroCoord.pop(0)
                                    heroCoord.append(res[0])
                                else:
                                    heroCoord.pop(0)
                                    heroCoord.append([])
                                    # elif len(heroCoord) > 0:
                                    #     temp_iou = iou.IoU(res, heroCoord)
                                    #     if temp_iou < 0.5:
                                    im0[res[0, 1]:res[0, 3], res[0, 0]:res[0, 2], :] = np.where(mask_rgb > 0, img_r,
                                                                                                im0_crop)
                                #     else:
                                #         heroCoord = res[0]

                            # im0_crop_recog = im0[res[0, 1]-20:res[0, 3]+20, res[0, 0]-20:res[0, 2]+20, :]

                            if 'person' in label and opt.person and count != max_idx:
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                        else:
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                        # print max_bbox midpoint
                        if count == max_idx:
                            mid = ((res[0, 2] + res[0, 0]) // 2, (res[0, 3] + res[0, 1]) // 2)

                    count += 1

            # cv2.imshow('im0', im0)
            # cv2.waitKey(0)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(0)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--person', action='store_true', help='displays only person')
    parser.add_argument('--heads', action='store_true', help='displays only person')
    opt = parser.parse_args()
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()


