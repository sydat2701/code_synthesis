import numpy as np 
import os
import cv2

import os
import cv2
import numpy as np
import json


def ordLabel(x):
	if x=='thump up':
		return 0
	if x=='ok':
		return 1
	if x=='lucky flower':
		return 2
	if x=='chinese hello':
		return 3
	if x=="bird":
		return 4
	if x=='take photo':
		return 5
	if x=='hand up':
		return 6
	if x=='hand down':
		return 7
	if x=='i love you':
		return 8
	if x=='flower':
		return 9
	if x=='muscle':
		return 10
	if x=='cross hand over hand':
		return 11
	if x=='hi':
		return 12
	if x=='stop':
		return 13
	if x=='rock':
		return 14
	if x=='sleepy':
		return 15
	if x=='pray':
		return 16
	if x=='heart':
		return 17
	if x=='big heart':
		return 18
	if x=='small heart':
		return 19
	if x=='binocullar':
		return 20
	if x=='plus sign':
		return 21
	if x=='dab':
		return 22
	if x=='shut up':
		return 23
	if x=='rabbit':
		return 24
	if x=='pistol':
		return 25
	if x=='hand over head':
		return 26
	if x=='touch cheek':
		return 27
	if x=='touch head':
		return 28
	if x=='wait':
		return 29
	if x=='calling':
		return 30
	if x=='no_gesture' or x=='no gesture':
		return 31


json_dir='/home/linhdt/Desktop/sydat/data.json'
with open(json_dir, "r") as fin:
	data=json.load(fin)

#data_des='C:\\Users\\ADMIN\\Desktop\\X_train.txt'
#data_des_y='C:\\Users\\ADMIN\\Desktop\\Y_train.txt'

train_X_dir="/home/linhdt/Desktop/sydat/datasetKP/X_train.txt"
train_Y_dir="/home/linhdt/Desktop/sydat/datasetKP/Y_train.txt"
val_X_dir="/home/linhdt/Desktop/sydat/datasetKP/X_val.txt"
val_Y_dir="/home/linhdt/Desktop/sydat/datasetKP/Y_val.txt"


data_X_test='/home/linhdt/Desktop/sydat/datasetKP/X_test.txt'
data_Y_test='/home/linhdt/Desktop/sydat/datasetKP/Y_test.txt'
data_X_test_2='/home/linhdt/Desktop/sydat/datasetKP/X_test2.txt'
data_Y_test_2='/home/linhdt/Desktop/sydat/datasetKP/Y_test2.txt'



cnt=0
dem1=0
count =0
last_item = []
tg =0
bh=0
for item in data:
	#print(item)		#item là key
	# try:

	# 	dem=item.count('/',0, len(item))
	# 	if dem<4:
	# 		continue
	# 	image_path= data_dir+ '/'+ item.split('/')[1].split('.')[1] + '/'+item.split('/')[2]+ '/'+item.split('/')[3] + '/'+item.split('/')[4]
		
	# except:
	# 	print("-------------------")
	# 	print(item.split('/')[1].split('.')[1])
	# 	print(item.split('/')[2])
	# 	print(item.split('/')[3])
		
	# dem=item.count('/',0, len(item))
	# if dem!=4:
	# 	print(item)

	tmp= data.get(item)  #value (hơi khác vs dictionary)
	#size = tmp['size']
	#tg1 = int(size[0])
	#tg2= int(size[1])
	#img=cv2.resize(img, (tg1, tg2))

	name_sub=item.split('/')[2].lower()

	# if name_sub=='subject_trinhgiang':
	# 	tg +=1
	# if name_sub=='subject_ban_hieu':
	# 	bh += 1

		#displayImg(img)
	
	tmp1=tmp['keypoints']
	for item1 in tmp1:
		set_type= item.split('/')[0].split('_')[1].lower()
		if set_type=='train' and (name_sub != 'subject_thugiang' and name_sub != 'subject_minh' and name_sub !='subject_trung' and name_sub!='subject_hangphung'):
			f =open(train_X_dir, "a")
		elif set_type=='train' and (name_sub == 'subject_thugiang' or name_sub == 'subject_minh'):
			f=open(val_X_dir,"a")
		elif set_type=='train' and (name_sub == 'subject_trung' or name_sub == 'subject_hangphung'):
			f=open(data_X_test_2,"a")
		elif set_type=='test':
			f=open(data_X_test, "a")
		for item2 in item1:
			cnt += 1
			if (cnt<54):
				continue
			if cnt > 54 and cnt <92:
				continue

			
			dem1 +=1
			if dem1<43:
				f.write(str(int(item2[0]))+', ')
				f.write(str(int(item2[1]))+', ')

			elif dem1==43:
				f.write(str(int(item2[0]))+', ')
				f.write(str(int(item2[1]))+'\n')

		#f.write('\n')
		f.close()

	name_ges=(item.split('/')[1].split('.')[1]).lower()
	#print(name_ges)
	# with open(data_des_y, "r+") as writer1:

	#f1=open(data_des_y, "a")
	set_type= item.split('/')[0].split('_')[1].lower()
	if set_type=='train' and name_sub != 'subject_thugiang' and name_sub != 'subject_minh' and name_sub !='subject_trung' and name_sub!='subject_hangphung':
		f1 =open(train_Y_dir, "a")
	elif set_type=='train' and (name_sub == 'subject_thugiang' or name_sub == 'subject_minh'):
		f1=open(val_Y_dir, "a")
	elif set_type=='train' and (name_sub == 'subject_trung' or name_sub == 'subject_hangphung'):
		f1=open(data_Y_test_2,"a")

	elif set_type=='test':
		f1=open(data_Y_test, "a")


	if type(ordLabel(name_ges)) != int:
		print(item)
	f1.write(str(ordLabel(name_ges))+'\n')
		


		#cv2.imwrite('D:\\Download\\output static\\'+ str(item.split('/')[1].split('.')[1]) +'_'+ str(item.split('/')[2])+str(count)+'.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])
	# 	count += 1
	cnt=0
	dem1=0


	