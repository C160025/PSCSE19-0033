# https://raw.githubusercontent.com/chia56028/Color-Transfer-between-Images/master/color_transfer.py

import numpy as np
import cv2
import os

def read_file(sn,tn):
	s = cv2.imread('images/source/'+sn+'.png')
	s = cv2.cvtColor(s,cv2.COLOR_BGR2LAB)
	t = cv2.imread('images/target/'+tn+'.png')
	t = cv2.cvtColor(t,cv2.COLOR_BGR2LAB)
	return s, t

def get_mean_and_std(x):
	x_mean, x_std = cv2.meanStdDev(x)
	x_mean = np.hstack(np.around(x_mean,2))
	x_std = np.hstack(np.around(x_std,2))
	return x_mean, x_std

def color_transfer():
	sources = ['sor3','sor4']#,'sor3.png','sor4.png','s5','s6']
	targets = ['tar3','tar4']#,'t3','t4','t5','t6']

	for n in range(len(sources)):
		print("Converting picture"+str(n+1)+"...")
		s, t = read_file(sources[n],targets[n])
		s_mean, s_std = get_mean_and_std(s)
		t_mean, t_std = get_mean_and_std(t)

		height, width, channel = s.shape
		for i in range(0,height):
			for j in range(0,width):
				for k in range(0,channel):
					x = s[i,j,k]
					x = ((x-s_mean[k])*(t_std[k]/s_std[k]))+t_mean[k]
					# round or +0.5
					x = round(x)
					# boundary check
					x = 0 if x<0 else x
					x = 255 if x>255 else x
					s[i,j,k] = x

		s = cv2.cvtColor(s,cv2.COLOR_LAB2BGR)
		cv2.imwrite('images/result/res2'+str(n+1)+'.png',s)

color_transfer()
os.system("pause")