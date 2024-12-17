import d4rl_atari
from utils import OfflineEnvAtari
import cv2
import numpy as np

dataset_path = '/home/rishav/scratch/d4rl_dataset/Seaquest/1/10'
data = OfflineEnvAtari(stack=False, path=dataset_path).get_dataset() 


#set 255 for all pixels > 0 and 0 for all pixels = 0
avg_img = np.mean(data['observations'][0] )
img_bw = np.where(data['observations'][0].transpose(1, 2, 0).squeeze(-1) > avg_img, 255, 0).astype(np.uint8)
#normalize the original image





cv2.imwrite('test.png', data['observations'][0].transpose(1, 2, 0).squeeze(-1))
cv2.imwrite('test_bw.png', img_bw)