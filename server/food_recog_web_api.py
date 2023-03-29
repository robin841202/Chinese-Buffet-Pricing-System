import cv2
import numpy as np
import easygui
import math
from flask import Flask, jsonify, request
import base64
import os
import json
import numpy as np
import cv2
import logging
import glob
from datetime import datetime
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import MySQLdb

dbconn=MySQLdb.connect(host="localhost",user="e508", passwd="f107118109", db="foods_recog", charset="utf8", port=3306)

app = Flask(__name__)

def loadModel():
    print('loading model...')
    global food_recog_model
    food_recog_model = load_model("models/foods_cate22_confirmed_299_aug_over70samples_InceptionV40.01lr_auto_k_fold_best.h5")
    food_recog_model._make_predict_function() #have to initialize before threading, let predict more faster
    print('loaded complete.')

def preprocess_inceptionv4(img, img_size=299):
	img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
	img = np.expand_dims(img, axis=0)
	img = img.astype("float32")/255.0
	return img

def preprocess_VGG16(img, img_size=224):
	img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
	img = np.expand_dims(img, axis=0)
	img = img.astype("float32")
	img = preprocess_input(img)
	return img

def intersection(line1, line2):
	"""Finds the intersection of two lines by using formula rho=xcos(theta)+ysin(theta).

	Returns closest integer pixel locations.
	See https://stackoverflow.com/a/383527/5087436
	"""
	rho1, theta1 = line1[0]
	rho2, theta2 = line2[0]
	A = np.array([
		[np.cos(theta1), np.sin(theta1)],
		[np.cos(theta2), np.sin(theta2)]
		])
	b = np.array([[rho1], [rho2]])
	x0, y0 = np.linalg.solve(A, b)
	x0, y0 = int(np.round(x0)), int(np.round(y0))
	return [x0, y0]

def drawLine(img,rho,theta):
	a = np.cos(theta)
	b = np.sin(theta)

	#公式: x = rho*cos(theta) , y = rho*sin(theta)
	x0 = a*rho #rho*cos(theta)
	y0 = b*rho #rho*sin(theta)

	x1 = int(x0 + 1000*(-b)) #(rho*cos(theta)-1000sin(theta))
	y1 = int(y0 + 1000*(a)) #(rho*sin(theta)+1000cos(theta))

	x2 = int(x0 - 1000*(-b)) #(rho*cos(theta)+1000sin(theta))
	y2 = int(y0 - 1000*(a)) #(rho*sin(theta)-1000cos(theta))

	cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2) #將霍夫轉換偵測到的直線延長畫滿整個影像

###從底盤中心點開始向上逐一pixel尋找上邊線###
def findTopLine(img_cp, ROI_centerX, ROI_centerY, interestedlines):
	for y in range(ROI_centerY,-1,-1):
		# print(y)
		# cv2.circle(img_cp, (ROI_centerX,int(y)), 3,(0,0,0), 1)
		for line in interestedlines:
			rho, theta = line[0]
			temp = ROI_centerX*np.cos(theta)+y*np.sin(theta)
			if ( np.round(temp) == np.round(rho)):
				return line
###從底盤中心點開始向與上邊線平行方向往右逐一pixel尋找右邊線###
def findRightLine(img_cp, ROI_centerX, ROI_centerY, theta_slope, interestedlines):
	rho_slope = ROI_centerX*np.cos(theta_slope) + ROI_centerY*np.sin(theta_slope)
	for x in range(ROI_centerX,img_cp.shape[1]+1):
		y = np.round((rho_slope-x*np.cos(theta_slope))/np.sin(theta_slope))
		#print(x, y)
		# cv2.circle(img_cp, (x,int(y)), 3,(255,0,0), 1)
		for line in interestedlines:
			rho, theta = line[0]
			temp = x*np.cos(theta)+y*np.sin(theta)
			if ( np.round(temp) == np.round(rho)):
				return line
###從底盤中心點開始向與上邊線平行方向往左逐一pixel尋找左邊線###
def findLeftLine(img_cp, ROI_centerX, ROI_centerY, theta_slope, interestedlines):
	rho_slope = ROI_centerX*np.cos(theta_slope) + ROI_centerY*np.sin(theta_slope)
	for x in range(ROI_centerX,-1,-1):
		y = np.round((rho_slope-x*np.cos(theta_slope))/np.sin(theta_slope))
		#print(x, y)
		# cv2.circle(img_cp, (x,int(y)), 3,(0,255,0), 1)
		for line in interestedlines:
			rho, theta = line[0]
			temp = x*np.cos(theta)+y*np.sin(theta)
			if ( np.round(temp) == np.round(rho)):
				return line
###從底盤中心點開始向與上邊線垂直方向往下逐一pixel尋找下邊線###
def findBottomLine(img_cp, ROI_centerX, ROI_centerY, theta_slope, interestedlines):
	rho_slope = ROI_centerX*np.cos(theta_slope) + ROI_centerY*np.sin(theta_slope)
	for y in range(ROI_centerY,img_cp.shape[0]+1):
		x = np.round((rho_slope-y*np.sin(theta_slope))/np.cos(theta_slope))
		#print(x, y)
		# cv2.circle(img_cp, (int(x),y), 3,(0,0,255), 1)
		for line in interestedlines:
			rho, theta = line[0]
			temp = x*np.cos(theta)+y*np.sin(theta)
			if ( np.round(temp) == np.round(rho)):
				return line


def restoreToOriginCoordinate(x, y, objectAngle, cx, cy, plate_x, plate_y, roughROI_x, roughROI_y):

	###還原回未旋轉前的座標###
	x_origin_cropped = int((x - cx) * math.cos(objectAngle) - (y - cy) * math.sin(objectAngle) + cx)
	y_origin_cropped = int((x - cx) * math.sin(objectAngle) + (y - cy) * math.cos(objectAngle) + cy)

	###還原回粗略ROI時的座標###
	x_origin_roughROI = x_origin_cropped+(plate_x-10)
	y_origin_roughROI = y_origin_cropped+(plate_y-10)

	###還原回原圖座標###
	x_origin = x_origin_roughROI+roughROI_x
	y_origin = y_origin_roughROI+roughROI_y

	return (x_origin, y_origin)
'''
###將多邊形所有座標還原回原圖座標###
def getPolyAllOriginCoordinates(mask, objectAngle, cx, cy, plate_x, plate_y, roughROI_x, roughROI_y):
	list_of_points_indices=np.nonzero(mask)
	areapointslist = []
	(y_list, x_list) = list_of_points_indices
	for i in range(len(x_list)):
		(x_origin, y_origin) = restoreToOriginCoordinate(x_list[i],y_list[i], objectAngle, cx,cy, plate_x,plate_y, roughROI_x, roughROI_y)
		areapointslist.append((x_origin, y_origin))
	return areapointslist
'''
def restoreListToOriginCoordinates(areapointslist, objectAngle, cx, cy, plate_x, plate_y, roughROI_x, roughROI_y):
	resultpointslist = []
	for (x,y) in areapointslist:
		(x_origin, y_origin) = restoreToOriginCoordinate(x,y, objectAngle, cx,cy, plate_x,plate_y, roughROI_x, roughROI_y)
		resultpointslist.append((x_origin, y_origin))
	return resultpointslist

def preprocess(img,emptyplate_init=False):
	origin_img = img.copy()
	#roughROI = [700, 1400, 450, 1000] #old roughROI
	roughROI = [700, 1400, 250, 800]
	img = img[roughROI[2]:roughROI[3],roughROI[0]:roughROI[1]]
	img_cp=img.copy()
	img_cp2=img.copy()

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #轉為灰階影像
	gray = cv2.GaussianBlur(gray,(3,3),0) #高斯模糊kernel3*3
	minthreshold = 300
	ratio = 3
	edges = cv2.Canny(gray,minthreshold,minthreshold*ratio,apertureSize = 3) #Canny邊緣檢測
	# cv2.imshow("Canny", edges)
	# cv2.waitKey(0)
	# 公式: rho = x*cos(theta)+y*sin(theta)
	# 公式: x = rho*cos(theta) , y = rho*sin(theta)
	lines = cv2.HoughLines(edges,1,np.pi/90,50) #霍夫轉換
	# print("HoughLines_length: ", len(lines))

	###計算所有初步偵測到的直線的交角並篩選出介於85~95度間的直線###
	interestedlines=[]
	temp=[]
	for n1 in range(len(lines)):
		for n2 in range(n1+1,len(lines)):
			#print(n1,n2)
			theta1 = lines[n1][0][1]
			theta2 = lines[n2][0][1]
			diff = abs(theta1-theta2)
			#print("相差幾度: ", math.degrees(diff))
			if np.isclose(math.degrees(diff), 90, atol=5):
				if n1 not in temp:
					interestedlines.append(lines[n1])
					temp.append(n1)
				if n2 not in temp:
					interestedlines.append(lines[n2])
					temp.append(n2)

	# print("interestedlines_length: ", len(interestedlines))

	# for line in interestedlines:
	# 	for r,theta in line:
	# 		print(r)
	# 		print(math.degrees(theta))
	# 		drawLine(img, r, theta)
	# cv2.imshow("img",img)
	# cv2.waitKey(0)


	###黑色底盤中心點座標###
	ROI_centerX = 360
	ROI_centerY = 235

	###找出餐盤上邊線###
	top_line = findTopLine(img_cp, ROI_centerX,ROI_centerY, interestedlines)
	top_rho,top_theta = top_line[0]
	# drawLine(img_cp, top_rho,top_theta)
	# cv2.imshow("top_line",img_cp)
	# cv2.waitKey(0)

	###找出餐盤右邊線###
	theta_horizontal = top_theta #平行theta一樣
	right_line = findRightLine(img_cp, ROI_centerX,ROI_centerY,theta_horizontal, interestedlines)
	right_rho, right_theta = right_line[0]
	# drawLine(img_cp, right_rho, right_theta)
	# cv2.imshow("right_line",img_cp)
	# cv2.waitKey(0)

	###找出餐盤左邊線###
	left_line = findLeftLine(img_cp, ROI_centerX,ROI_centerY,theta_horizontal, interestedlines)
	left_rho, left_theta = left_line[0]
	# drawLine(img_cp, left_rho, left_theta)
	# cv2.imshow("left_line", img_cp)
	# cv2.waitKey(0)

	###找出餐盤下邊線###
	theta_vertical = top_theta + np.pi/2
	bottom_line = findBottomLine(img_cp, ROI_centerX,ROI_centerY,theta_vertical, interestedlines)
	bottom_rho, bottom_theta = bottom_line[0]
	# drawLine(img_cp, bottom_rho, bottom_theta)
	# cv2.imshow("bottom_line", img_cp)
	# cv2.waitKey(0)

	###計算出4個角點並找出boundingRect在周遭保留10像素裁切影像###
	LT_point = intersection(top_line,left_line)
	RT_point = intersection(top_line,right_line)
	RB_point = intersection(bottom_line,right_line)
	LB_point = intersection(bottom_line,left_line)
	points = np.array([[LT_point,RT_point,RB_point,LB_point]])
	(plate_x, plate_y, w, h) = cv2.boundingRect(points)
	cropped_img = img_cp2[plate_y-10:plate_y+h+10,plate_x-10:plate_x+w+10]
	# cv2.imshow("cropped_img", cropped_img)
	# cv2.waitKey(0)

	cropped_width = cropped_img.shape[1]
	cropped_height = cropped_img.shape[0]
	###座標轉換成cropped過後的###
	LT_point_cropped = [LT_point[0]-(plate_x-10),LT_point[1]-(plate_y-10)]
	RT_point_cropped = [RT_point[0]-(plate_x-10),RT_point[1]-(plate_y-10)]
	RB_point_cropped = [RB_point[0]-(plate_x-10),RB_point[1]-(plate_y-10)]
	LB_point_cropped = [LB_point[0]-(plate_x-10),LB_point[1]-(plate_y-10)]
	points_cropped = np.array([[LT_point_cropped,RT_point_cropped,RB_point_cropped,LB_point_cropped]])

	###利用4個角點畫出餐盤contour###
	gray_contour_img = np.zeros((cropped_height, cropped_width), dtype=np.uint8)
	cv2.drawContours(gray_contour_img,points_cropped, -1, 255, 1)
	ret,binary_contour_img = cv2.threshold(gray_contour_img,127,255,0)
	# cv2.imshow("gray_contour_img", binary_contour_img)
	# cv2.waitKey(0)


	# centroid = cropped_img.copy()

	###計算質心###
	M = cv2.moments(points_cropped)
	cx = int(M["m10"]/M["m00"])
	cy = int(M["m01"]/M["m00"])
	# print("centerX: ", cx)
	# print("centerY: ", cy)
	# cv2.circle(centroid, (cx, cy), 5, (0, 0, 255), -1)
	# cv2.imshow("contours result", fill_result)
	# cv2.waitKey(0)
	# cv2.imshow("centroid", centroid)
	# cv2.waitKey(0)

	###原先使用的影像矩角度正規化法，但由於餐盤contour由霍夫直線的四角點取得(有些微誤差)，會造成餐盤長軸推算誤差, 因此最後直接使用將下邊線旋轉至與x軸平行較準確###
	# u11=0
	# u20=0
	# u02=0
	###透過原始矩推算中心矩u11...可看維基百科圖像矩公式###
	# u11 = (M["m11"]/M["m00"]) - (cx*cy)
	# u20 = (M["m20"]/M["m00"]) - (cx*cx)
	# u02 = (M["m02"]/M["m00"]) - (cy*cy)
	# print("u11: ", u11)
	# print("u02: ", u02)
	# print("u20: ", u20)
	# testnum=(2*u11)/(u20-u02)
	# objectAngle=0.5*math.atan(testnum)
	# print("objA:",objectAngle)
	# angle = math.degrees(objectAngle)
	# print("Angle:",angle)
	###畫出長軸並秀出###
	# def drawMajorAxisPoint(r,cx,cy,angle,color):
	# 	dx = r*np.cos(angle)
	# 	dy = r*np.sin(angle)
	# 	cv2.line(centroid, (cx,cy),(int(cx-dx),int(cy-dy)),color,3)
	# 	cv2.imshow("majorAxis",centroid)
	# 	cv2.waitKey(0)
	# drawMajorAxisPoint(200,cx,cy,objectAngle,(0,255,0)) #畫出長軸
	# drawMajorAxisPoint(200,cx,cy,0,(0,0,255)) #畫出與x軸平行經質心的線
	# m = cv2.getRotationMatrix2D((cx,cy),(int)(angle),1.0)


	###利用餐盤下邊線做角度正規化###
	#print("bottom_theta",math.degrees(bottom_theta))
	rotate_angle = -(int)(180-90-math.degrees(bottom_theta))
	m = cv2.getRotationMatrix2D((cx,cy),rotate_angle,1.0)
	rotated = cv2.warpAffine(cropped_img, m, (cropped_width, cropped_height))
	###秀出示意圖###
	# rotated_show = cv2.warpAffine(centroid, m, (cropped_width, cropped_height))
	# cv2.imwrite("pics/temp_crop/cropped_img.jpg",cropped_img)
	# cv2.imwrite("pics/temp_crop/rotated2.jpg",rotated_show)
	# cv2.imshow("rotated",rotated_show)
	# cv2.waitKey(0)


	###以質心為基準點算出盤子內各個菜區的座標，並使用遮罩裁剪下來###
	mask1 = np.zeros((cropped_height, cropped_width), dtype=np.uint8)
	mask2 = np.zeros((cropped_height, cropped_width), dtype=np.uint8)
	mask3 = np.zeros((cropped_height, cropped_width), dtype=np.uint8)
	mask4 = np.zeros((cropped_height, cropped_width), dtype=np.uint8)
	points1 = np.array([[[cx-141,cy-113],[cx-65,cy-113],[cx-40,cy-12],[cx-141,cy+49]]])
	points2 = np.array([[[cx-62,cy-113],[cx+68,cy-113],[cx+36,cy-12],[cx-38,cy-12]]])
	points3 = np.array([[[cx+70,cy-113],[cx+136,cy-113],[cx+136,cy+49],[cx+41,cy-11]]])
	points4 = np.array([[[cx-141,cy+60],[cx-38,cy-5],[cx+35,cy-5],[cx+136,cy+59],[cx+136,cy+112],[cx-141,cy+112]]])
	###產生mask###
	cv2.fillPoly(mask1, points1, (255))
	cv2.fillPoly(mask2, points2, (255))
	cv2.fillPoly(mask3, points3, (255))
	cv2.fillPoly(mask4, points4, (255))
	###使用mask裁切菜區ROI###
	result1 = cv2.bitwise_and(rotated,rotated,mask = mask1)
	result1[mask1==0] = (255,255,255)
	result2 = cv2.bitwise_and(rotated,rotated,mask = mask2)
	result2[mask2==0] = (255,255,255)
	result3 = cv2.bitwise_and(rotated,rotated,mask = mask3)
	result3[mask3==0] = (255,255,255) 
	result4 = cv2.bitwise_and(rotated,rotated,mask = mask4)
	result4[mask4==0] = (255,255,255) 
	rect1 = cv2.boundingRect(points1) # returns (x,y,w,h) of the rect
	rect2 = cv2.boundingRect(points2) # returns (x,y,w,h) of the rect
	rect3 = cv2.boundingRect(points3) # returns (x,y,w,h) of the rect
	rect4 = cv2.boundingRect(points4) # returns (x,y,w,h) of the rect
	cropped1 = result1[rect1[1]: rect1[1] + rect1[3], rect1[0]: rect1[0] + rect1[2]]
	cropped2 = result2[rect2[1]: rect2[1] + rect2[3], rect2[0]: rect2[0] + rect2[2]]
	cropped3 = result3[rect3[1]: rect3[1] + rect3[3], rect3[0]: rect3[0] + rect3[2]]
	cropped4 = result4[rect4[1]: rect4[1] + rect4[3], rect4[0]: rect4[0] + rect4[2]]

	###校正空盤時的菜區取較小 避免周遭凸起處影響深度平均###
	if emptyplate_init == True:
		scale = 25
		area1pointslist = [(cx-141+scale,cy-113+scale),(cx-65-scale,cy-113+scale),(cx-40-scale,cy-12-scale),(cx-141+scale,cy+49-scale)]
		area2pointslist = [(cx-62+scale,cy-113+scale),(cx+68-scale,cy-113+scale),(cx+36-scale,cy-12-scale),(cx-38+scale,cy-12-scale)]
		area3pointslist = [(cx+70+scale,cy-113+scale),(cx+136-scale,cy-113+scale),(cx+136-scale,cy+49-scale),(cx+41+scale,cy-11-scale)]
		area4pointslist = [(cx-141+scale,cy+60+scale),(cx-38+scale,cy-5+scale),(cx+35-scale,cy-5+scale),(cx+136-scale,cy+59+scale),(cx+136-scale,cy+112-scale),(cx-141+scale,cy+112-scale)]
	else:
		scale = 15
		area1pointslist = [(cx-141+scale,cy-113+scale),(cx-65-scale,cy-113+scale),(cx-40-scale,cy-12-scale),(cx-141+scale,cy+49-scale)]
		area2pointslist = [(cx-62+scale,cy-113+scale),(cx+68-scale,cy-113+scale),(cx+36-scale,cy-12-scale),(cx-38+scale,cy-12-scale)]
		area3pointslist = [(cx+70+scale,cy-113+scale),(cx+136-scale,cy-113+scale),(cx+136-scale,cy+49-scale),(cx+41+scale,cy-11-scale)]
		area4pointslist = [(cx-141+scale,cy+60+scale),(cx-38+scale,cy-5+scale),(cx+35-scale,cy-5+scale),(cx+136-scale,cy+59+scale),(cx+136-scale,cy+112-scale),(cx-141+scale,cy+112-scale)]

	###將所有菜區角點還原回原圖座標用於回傳給client端做影像座標對映###
	area1pointslist = restoreListToOriginCoordinates(area1pointslist, math.radians(rotate_angle), cx,cy, plate_x,plate_y, roughROI[0], roughROI[2])
	area2pointslist = restoreListToOriginCoordinates(area2pointslist, math.radians(rotate_angle), cx,cy, plate_x,plate_y, roughROI[0], roughROI[2])
	area3pointslist = restoreListToOriginCoordinates(area3pointslist, math.radians(rotate_angle), cx,cy, plate_x,plate_y, roughROI[0], roughROI[2])
	area4pointslist = restoreListToOriginCoordinates(area4pointslist, math.radians(rotate_angle), cx,cy, plate_x,plate_y, roughROI[0], roughROI[2])

	
	if emptyplate_init == True:
		return [area1pointslist, area2pointslist, area3pointslist, area4pointslist]
	else:
		return( [cropped1, cropped2, cropped3, cropped4], [area1pointslist, area2pointslist, area3pointslist, area4pointslist])

@app.route("/api/recognize", methods=['POST'])
def recognize():
	print("recognizing...")
	request_data = request.json
	food_data = request_data['FoodPic']
	if food_data != None:
		###base64解碼並轉回opencv影像###
		b64decodedImage = base64.b64decode(food_data)
		nparr = np.fromstring(b64decodedImage, dtype=np.uint8)
		food_image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)

		(cropped_img_lists,arealists) = preprocess(food_image) #裁切出菜區

		food_resultlist = []
		price_list = []
		calories_per_g_list = []
		volume_per_portion_list = []
		for index, cropped_img in enumerate(cropped_img_lists):
			preprocessed_img = preprocess_inceptionv4(cropped_img)
			prediction = food_recog_model.predict(preprocessed_img)
			pred_idx = np.argmax(prediction, axis=1)
			highest_score = prediction[0][pred_idx]
			result = pred_idx[0]
			print("result{0}: {1}".format(index, pred_idx[0]))
			print("result{0} score: {1}".format(index, highest_score))

			###查詢資料庫表###
			try:
				cursor=dbconn.cursor()
				SQL_query = "SELECT food_name, price, calories_per_g, volume_per_portion FROM foods_table WHERE id = %(id)s"
				cursor.execute(SQL_query, {"id":result+1})
				food_name, price, calories_per_g, volume_per_portion = cursor.fetchone()
				food_resultlist.append(food_name)
				price_list.append(price)
				calories_per_g_list.append(calories_per_g)
				volume_per_portion_list.append(volume_per_portion)
			except MySQLdb.Error:
				logging.warn("Failed to query.")
				dbconn.rollback()
			finally:
				cursor.close()

		ROI_result = {"area1": arealists[0], "area2": arealists[1], "area3": arealists[2], "area4": arealists[3]}
		food_result = {"result1": food_resultlist[0], "result2": food_resultlist[1], "result3": food_resultlist[2], "result4": food_resultlist[3]}
		food_price = {"price1": price_list[0], "price2": price_list[1], "price3": price_list[2], "price4": price_list[3]}
		food_calories = {"calories1": calories_per_g_list[0], "calories2": calories_per_g_list[1], "calories3": calories_per_g_list[2], "calories4": calories_per_g_list[3]}
		food_volume_per_portion = {"volume1": volume_per_portion_list[0], "volume2": volume_per_portion_list[1], "volume3": volume_per_portion_list[2], "volume4": volume_per_portion_list[3]}
	else:
		ROI_result = None
		food_result = None
		food_price = None
		food_calories = None
		food_volume_per_portion = None
	response = {"ROI": ROI_result, "FoodResult": food_result, "FoodPrices": food_price, "FoodCalories": food_calories, "FoodVolumePerPortion": food_volume_per_portion}
	return jsonify(response) 


@app.route("/api/emptyplate_init", methods=['POST'])
def emptyplate_init():
	print("processing...")
	request_data = request.json
	food_data = request_data['FoodPic']
	if food_data != None:
		b64decodedImage = base64.b64decode(food_data)
		nparr = np.fromstring(b64decodedImage, dtype=np.uint8)
		food_image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
		arealists = preprocess(food_image, True)
		result = {"area1": arealists[0], "area2": arealists[1], "area3": arealists[2], "area4": arealists[3]}
	else:
		result = None
	response = {"ROI": result}
	return jsonify(response) 

if __name__ == "__main__":
	loadModel()
	app.run(host='localhost', port=5000)