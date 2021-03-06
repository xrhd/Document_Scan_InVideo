'''Richard HD'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave


from scan.utils import *
from scan.video import WebcamVideoStreamer, VideoStreamer


### STREAMING CAPITURES  ###


def scan_on_streaming(n_samples=1000):
    """
    This function starts the screaming object;
    And from each frame tries to extract the document scan;
    """
    print('[INFO] Starting video stream...')
    vs = VideoStreamer(src=0).start()
    time.sleep(0.2)

    '''start the FPS throughput estimator'''
    fps = FPS().start()
    encodings = []
    labelSet = set()


    '''loop over frames from the video file stream'''
    scans = list()
    try:
        while True:
            frame = vs.read() # grab the frame from the threaded video stream
           
            """SCAN FROM FRAME"""
            scaned = scan_from_frame(frame)
            if np.any(scaned):
                if len(scans)==n_samples: break
                scans += [scaned]
            
            '''update the FPS counter'''
            fps.update()
            cv2.imshow("Frame", frame)
            
            '''stop scan'''
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break

    except: pass
    finally:
        '''Stop Some processes''' 
        fps.stop()
        cv2.destroyAllWindows()
        vs.stop()

    '''feed back'''
    global mean_fps
    mean_fps = fps.fps()
    print(f'[INFO] elasped time: {fps.elapsed():.2f}')
    print(f'[INFO] approx. FPS: {fps.fps():.2f}')
    return scans


### MAIN ###

if __name__ == "__main__":
    '''scan documet from image'''
    scans = scan_on_streaming(n_samples=100)
    scans = dbscan_filter(scans , 30)

    '''get the better ones'''
    blurry = list(np.sort([cv2.Laplacian(img, cv2.CV_64F).var() for img in scans]))
    less_blurry = blurry[::-1][:5]

    '''plot the figures'''
    plt.figure(figsize=(15,10))
    plt.suptitle('less blurry images')
    for i,blurriness in enumerate(less_blurry):
        selected = scans[blurry.index(blurriness)]
        plt.subplot(1,5,(i+1))
        plt.title(f'Doc{i+1}')
        plt.imshow(selected, cmap='gray')

        '''save docs into image files'''
        imsave(f'DEMO/doc{i}.png', selected)

    plt.show()