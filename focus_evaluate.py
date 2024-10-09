# tobii gaze position retrieve
import json
import glob
import cv2
import os, sys
import gzip
import numpy as np
import dotenv
import argparse

dotenv.load_dotenv()

#def focus_eval(im):

# compute RMS contract of im
def RMS_contrast(im):
    img_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    contrast = img_grey.std()

    return contrast

# compute Michelson contrast
def Michelson_contrast(im):
    Y = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)[:,:,0]
    
    # compute min and max of Y
    min = np.min(Y)
    max = np.max(Y)

    # compute contrast
    contrast = (max-min)/(max+min)

    return contrast

if __name__ == '__main__':

    # コマンドライン引数を解釈する
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--file")
    args = parser.parse_args()

    if args.file is None:
        print("specify task by option -f [file name]")
        sys.exit(0)

    # read mp4 from video
    cap = cv2.VideoCapture(args.file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
    # region of interst(s)
    ROI = []
    CONTRAST = []

    # mouse callback
    def mousecallback(event,x,y,flags,param):
        global ROI

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(ROI) == 0:
                ROI.append([x,y])
            else:
                if len(ROI[-1]) == 2:
                    p = ROI.pop()
                    w = abs(x - p[0])
                    h = abs(y - p[1])
                    if p[0] < x:
                        x = p[0]
                    if p[1] < y:
                        y = p[1]
                    ROI.append([x,y,w,h])
                else:
                    ROI.append([x,y])

    # start processing
    n = 0
    PAUSE = False
    DATA = []

    while(True):
        ret, frame = cap.read()
        if frame is None:
            break

        if n > 0:
            ti = float(n)/float(fps)
        else:
            ti = 0

        # draw frame number
        cv2.putText(frame, f"frame {n}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255))

        # draw ROI
        for roi in ROI:
            cv2.rectangle(frame, (roi[0],roi[1]), (roi[0]+roi[2],roi[1]+roi[3]), (255,255,0), 1, 1)

        # compute contrast of ROIs
        CONTRAST = []
        for i,roi in enumerate(ROI):
            if len(roi) == 2:
                continue
            
            im = frame[roi[1]:(roi[1]+roi[3]),roi[0]:(roi[0]+roi[2]),:]
            
            # if render the cropped area
            im2 = cv2.resize(im, dsize=(512, 512))
            frame[0:512,(width-512):width] = im2
            CONTRAST.append([RMS_contrast(im), Michelson_contrast(im)])
            cv2.putText(frame, f"contrast {CONTRAST[-1][0]:2.3f} {CONTRAST[-1][1]:2.3f}", (20, 75), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255))
            DATA.append([n,i,CONTRAST[-1][0],CONTRAST[-1][1]])

        cv2.imshow('monitor', frame)
        cv2.setMouseCallback('monitor', mousecallback)

        if PAUSE:
            k = cv2.waitKey(0)
        else:
            k = cv2.waitKey(5)

        if k == 27:
            # ESC to quit
            cv2.destroyAllWindows()
            break

        if k == ord(' '):
            PAUSE = True if PAUSE is False else False

        n += 1

    # write to log file
    import csv

    with open(args.file + ".log","wt",newline='') as f:
        wri = csv.writer(f)
        wri.writerows(DATA)

    cap.release()
    cv2.destroyAllWindows() 