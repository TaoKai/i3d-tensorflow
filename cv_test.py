import cv2
import numpy as np

def testSampleRGB(path):
    video = np.load(path)[0]
    video = list(video)
    imgs = []
    for i, frame in enumerate(video):
        frame = (cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)+1)/2
        cv2.imshow('video', frame)
        cv2.waitKey(40)
        imgs.append(frame)
    return imgs

def testSampleFlow(path):
    video = np.load(path)[0]
    video = list(video)
    imgs = []
    for i, frame in enumerate(video):
        shp = frame.shape
        mat = np.zeros([shp[0], shp[1], 1], dtype=np.float32)
        frame = np.concatenate([frame, mat], axis=2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame += 0.5
        cv2.imshow('video', frame)
        cv2.waitKey(40)
        imgs.append(frame)
    return imgs


def compute_TVL1(prev, curr):
    """Compute the TV-L1 optical flow."""
    TVL1=cv2.optflow.DualTVL1OpticalFlow_create()
    # TVL1 = cv2.DualTVL1OpticalFlow_create()
    # TVL1=cv2.createOptFlow_DualTVL1()
    flow = TVL1.calc(prev, curr, None)
    # assert flow.dtype == np.float32
    # flow = (flow + bound) * (255.0 / (2 * bound))
    # flow = np.round(flow).astype(int)
    flow[flow >= 10] = 10
    flow[flow <= -10] = -10
 
    return flow


def getFlows(frames):
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    flows = []
    for i, f in enumerate(frames):
        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        flow = compute_TVL1(prev, f)
        mat = np.zeros([flow.shape[0], flow.shape[1], 1], np.float32)
        flowShow = np.concatenate([flow, mat], axis=2)/10 + 0.5
        flowShow = cv2.cvtColor(flowShow, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', flowShow)
        cv2.waitKey(1)
        flows.append(flowShow)
        prev = f
        print('collect flow', i+1, np.max(flow), np.min(flow))
    return flows


def getFrames(path):
    cap = cv2.VideoCapture(path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    rate = cap.get(cv2.CAP_PROP_FPS)
    print('total:', total_frames, 'Height:', height, 'Width:', width, 'Rate:', rate, 'Length:', rate*total_frames/1000)
    cnt = 1
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            print('collect frame', cnt)
            frames.append(frame)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            cnt += 1
        else:
            break
    cap.release()
    return frames



if __name__ == "__main__":
    path_rgb = 'data\\v_CricketShot_g04_c01_rgb.npy'
    path_flow = 'data\\v_CricketShot_g04_c01_flow.npy'
    vPath = "D:\\workspace\\data\\UCF11_updated_mpg\\tennis_swing\\v_tennis_01\\v_tennis_01_05.mpg"
    frames = getFrames(vPath)
    flows = getFlows(frames)
    for fr, fl in zip(frames, flows):
        f = np.concatenate([fr/255.0, fl], axis=1)
        cv2.imshow('frame', f)
        cv2.waitKey(40)