import cv2
import numpy as np
import sys, os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
LOCK = threading.Lock()
totalCnt = 0
rangeCnt = 0
ids_train = None
ids_test = None
test_path = 'data/test'
train_path = 'data/train'

def listVideos(path):
    print('get video clip list ...')
    dirs = os.listdir(path)
    id_map = {d:i for i, d in enumerate(dirs)}
    videoList = []
    for d in dirs:
        bp = os.path.join(path, d)
        avis = [(os.path.join(bp, v), id_map[d], d) for v in os.listdir(bp)]
        videoList += avis
    np.random.shuffle(videoList)
    return videoList


def randomSelect(frames, num=48):
    raw_frames = []
    for f in frames:
       if f.shape[0]==224 and f.shape[1]==224 and f.shape[2]==3:
           raw_frames.append(f)
    if len(raw_frames)<=num:
        return np.array(raw_frames, np.uint8)
    else:
        start = (len(raw_frames)-num)*np.random.random()
        start = int(start)
        return np.array(raw_frames[start:start+num], np.uint8)


def randomCrop(frames, size, ifRand=True):
    h = size[0]
    w = size[1]
    x = 0
    y = 0
    if not ifRand:
        x = int((w/2.0-112)/2)
        y = int((h/2.0-112)/2)
    else:
        h0 = h-224-1
        w0 = w-224-1
        x = int(w0*np.random.random())
        y = int(h0*np.random.random())
    framesMat = np.stack(frames).astype(np.uint8)
    framesMat = framesMat[:, y:y+224, x:x+224, :]
    return list(framesMat)

def calVideos(videoList):
    ratio = 5
    train = []
    test = []
    global rangeCnt, totalCnt, LOCK, ids_train, ids_test
    for i, v in enumerate(videoList):
        p = v[0]
        cap = cv2.VideoCapture(p)
        fCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fRate = cap.get(cv2.CAP_PROP_FPS)
        fLength = fCount*fRate
        fSample = fLength/40.0
        fIntv = fCount/fSample
        sFrames = sampleVideo(cap, fIntv)
        if fHeight<224 or fWidth<224:
            continue
        if len(sFrames)>=32:
            if i%ratio==0 and len(sFrames)>=64:
                frames = randomCrop(sFrames, (fHeight, fWidth), False)
                frames = randomSelect(frames, 64)
                if len(frames)<64:
                    continue
                # test.append((frames, v[1], v[2]))
                name = os.path.split(p)[-1].split('.')[0]
                LOCK.acquire()
                np.save(os.path.join(test_path, name+'.npy'), frames)
                test.append((name+'.npy', v[1], v[2]))
                rangeCnt += 1
                ids_test[v[2]] += 1
                totalCnt += 1
                print(totalCnt, v[2], fCount, fHeight, fWidth, fSample, fIntv)
                LOCK.release()
            else:
                frames = randomCrop(sFrames, (fHeight, fWidth), True)
                frames = randomSelect(frames, 32)
                if len(frames)<32:
                    continue
                # train.append((frames, v[1], v[2]))
                name = os.path.split(p)[-1].split('.')[0]
                LOCK.acquire()
                np.save(os.path.join(train_path, name+'.npy'), frames)
                train.append((name+'.npy', v[1], v[2]))
                rangeCnt += 1
                ids_train[v[2]] += 1
                totalCnt += 1
                print(totalCnt, v[2], fCount, fHeight, fWidth, fSample, fIntv)
                LOCK.release()
        cap.release()
        
        # simplePlay(frames, 40)
    return (train, test)


def simplePlay(frames, rate):
    for f in frames:
        cv2.imshow('play', f)
        cv2.waitKey(rate)


def sampleVideo(cap, interval):
    cnt = 0
    cntv = 0.0
    framesSample = []
    while True:
        ret, frame = cap.read()
        if ret:
            if cnt == round(cntv):
                frame = cv2.resize(frame, (320, 240), cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                framesSample.append(frame)
                cntv += interval
            cnt += 1
        else:
            break
    return framesSample


def multiProcess(videoList):
    def split_list(vList, num):
        if num <=1:
            return [vList]
        total_list = []

        vcount = len(vList)//(num-1)
        for i in range(num):
            sp_list = vList[i*vcount: (i+1)*vcount]
            total_list.append(sp_list)
        return total_list
    global ids_train, ids_test
    test_cnt = 0
    train_cnt = 0
    ids_train = {v[2]:0 for v in videoList}
    ids_test = ids_train.copy()
    total_list = split_list(videoList, 5)
    executor = ThreadPoolExecutor(max_workers=4)
    taskList = [executor.submit(calVideos, (tl)) for tl in total_list]
    train = []
    test = []
    for t in as_completed(taskList):
        res = t.result()
        train += res[0]
        test += res[1]
    label_path = 'data'
    np.save(os.path.join(label_path, 'train.npy'), train)
    np.save(os.path.join(label_path, 'test.npy'), test)
    for k, v in ids_train.items():
        print(k, v, ids_test[k])
        test_cnt += ids_test[k]
        train_cnt += v
    print('train:', train_cnt, 'test:', test_cnt)




if __name__ == "__main__":
    basePath = "../../data/UCF-101"
    vList = listVideos(basePath)
    multiProcess(vList)
    #ids_train = {v[2]:0 for v in vList}
    #ids_test = ids_train.copy()
    #calVideos(vList)
