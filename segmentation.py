# utility functions

import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.fftpack # For FFT2 


def cv2_imshow(img):
	cv2.imshow("img",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



def predict_from_model(image,model):
	image = cv2.resize(image,(80,80))
	image = np.stack((image,)*3, axis=-1)
	prediction = np.argmax(model.predict(image[np.newaxis,:]))
	return prediction
def four_point_transform(image, pts):

	rect = order_points(pts,image.shape)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped
def order_points(pts,shape):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[0] = [max(rect[0][0]-15,0),max(rect[0][1]-15,0)]
	rect[2] = pts[np.argmax(s)]
	rect[2] = [min(rect[2][0]+15,shape[1]-1),min(rect[2][1]+15,shape[0]-1)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[1] = [min(rect[1][0]+15,shape[1]-1),max(rect[1][1]-15,0)]

	rect[3] = pts[np.argmax(diff)]
	rect[3] = [max(rect[3][0]-15,0),min(rect[3][1]+15,shape[0]-1)]

	print(rect)
	return rect
# Now let's create a mask for this image
def createMask(size, hull):
	(rows, cols)  = size
	# black image
	mask = np.zeros((rows, cols), dtype=np.uint8)
	# blit our contours onto it in white color
	cv2.drawContours(mask, hull, 0, 255, -1)
	return mask
# Check if a rectangle present inside other
# helps is selecting the outermost rectangle
def rectContains(rect, pt):
	ret = rect[0] <= pt[0] <= rect[0] + rect[2] and rect[1] <= pt[1] <= rect[1] + rect[3]
	return ret

# Find all the biggest rectangles/contours in the image
def refine_contours(list_coord):
	ret_cnt = list_coord.copy()

	for i in list_coord:
		for j in list_coord:
			if rectContains(i, [j[0], j[1]]) and rectContains(i, [j[0] + j[2], j[1]]) and rectContains(i, [j[0], j[1] + j[3]]) and rectContains(i, [j[0] + j[2], j[1] + j[3]]) and i != j:
				if j in ret_cnt:
					ret_cnt.remove(j)
		
	return ret_cnt


def CreateHull(plate):
	gray_plate = cv2.cvtColor( plate , cv2.COLOR_BGR2GRAY )
	gray_plate = cv2.fastNlMeansDenoising(gray_plate , h= 3,templateWindowSize = 7,searchWindowSize = 21)
	# cv2_imshow(gray_plate)

	gray_plate = cv2.GaussianBlur(gray_plate, (3, 3), 0)

	# ret3,thresh = cv2.threshold(gray_plate,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# kernel = np.ones((7,7),np.uint8)
	# thresh = cv2.erode(thresh,kernel,iterations = 1)
	# cv2_imshow(thresh)#imp
	# plt.hist(gray_plate.ravel(),256,[0,256]); plt.show()
	pix = [] 
	for i in range(256):
		pix.append(0)
	for i in range(gray_plate.shape[0]):
		for j in range(gray_plate.shape[1]):
			pix[gray_plate[i][j]]+=1
	# print(pix)
	mx = max(pix)
	# print(max(pix))
	print(pix.index(mx))
	limit = pix.index(mx)
	thresh = gray_plate >= max(90,limit*0.58)
	thresh = 255*thresh.astype("uint8")
	# cv2_imshow(thresh)

	contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	final_cnt = None
	max_area = 0
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if(area>max_area):
			max_area = area
			final_cnt = cnt
	drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
	hull = []
	hull.append(cv2.convexHull(final_cnt, False))
	hull2 = cv2.convexHull(final_cnt, False)
	color = (255, 0, 0)
	cv2.drawContours(drawing, hull,0, color, 1, 8)
	# cv2_imshow(drawing)
	return hull


def imclearborder(imgBW, radius):

	# Given a black and white image, first find all of its contours
#	 cv2_imshow(imgBW)
	imgBWcopy = imgBW.copy()
	contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
		cv2.CHAIN_APPROX_SIMPLE)

	# Get dimensions of image
	imgRows = imgBW.shape[0]
	imgCols = imgBW.shape[1]	

	contourList = [] # ID list of contours that touch the border

	# For each contour...
	for idx in np.arange(len(contours)):
		# Get the i'th contour
		cnt = contours[idx]

		# Look at each point in the contour
		for pt in cnt:
			rowCnt = pt[0][1]
			colCnt = pt[0][0]

			# If this is within the radius of the border
			# this contour goes bye bye!
			check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
			check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

			if check1 or check2:
				contourList.append(idx)
				break

	for idx in contourList:
		cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)
#	 cv2_imshow(imgBW)
	return imgBWcopy


#### bwareaopen definition
def bwareaopen(imgBW, areaPixels):
	# Given a black and white image, first find all of its contours
	# cv2_imshow(imgBW)
	imgBWcopy = imgBW.copy()
	contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
		cv2.CHAIN_APPROX_SIMPLE)

	# For each contour, determine its total occupying area
	for idx in np.arange(len(contours)):
		# area = cv2.contourArea(contours[idx])
		x, y, w, h = cv2.boundingRect(contours[idx])
		area = w*h
		if (area >= 0 and area <= areaPixels):
	  		cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)
	# cv2_imshow(imgBWcopy)
	return imgBWcopy


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def refine_final(imgBW):
	# Given a black and white image, first find all of its contours
	imgBWcopy = imgBW.copy()
	contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
		cv2.CHAIN_APPROX_SIMPLE)

	# Get dimensions of image
	imgRows = imgBW.shape[0]
	imgCols = imgBW.shape[1] 
	for idx in np.arange(len(contours)):
		area = cv2.contourArea(contours[idx])
		if(area<imgRows*imgCols*0.02):
			cv2.drawContours(imgBW, contours, idx, (0,0,0), -1)
	return imgBW

def ClearThresold(wrap):
	img = wrap.copy()
	# img = cv2.resize(img,(6*img.shape[1],6*img.shape[0]))

	# Number of rows and columns
	rows = img.shape[0]
	cols = img.shape[1]

	# Remove some columns from the beginning and end
	# img = img[:, 59:cols-20]

	# Number of rows and columns
	rows = img.shape[0]
	cols = img.shape[1]

	# Convert image to 0 to 1, then do log(1 + I)
	imgLog = np.log1p(np.array(img, dtype="float") / 255)

	# Create Gaussian mask of sigma = 10
	M = 2*rows + 1
	N = 2*cols + 1
	sigma = 10
	(X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
	centerX = np.ceil(N/2)
	centerY = np.ceil(M/2)
	gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2

	# Low pass and high pass filters
	Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
	Hhigh = 1 - Hlow

	# Move origin of filters so that it's at the top left corner to
	# match with the input image
	HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
	HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

	# Filter the image and crop
	If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
	Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
	Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))

	# Set scaling factors and add
	gamma1 = 0.3
	gamma2 = 1.5
	Iout = gamma1*Ioutlow[0:rows,0:cols] + gamma2*Iouthigh[0:rows,0:cols]

	# Anti-log then rescale to [0,1]
	Ihmf = np.expm1(Iout)
	Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
	Ihmf2 = np.array(255*Ihmf, dtype="uint8")
	# cv2_imshow(Ihmf2)
	# plt.hist(Ihmf2.ravel(),256,[0,256]); plt.show()
	pix = [] 
	for i in range(256):
		pix.append(0)
	for i in range(Ihmf2.shape[0]):
		for j in range(Ihmf2.shape[1]):
			pix[Ihmf2[i][j]]+=1
	# print(pix)
	mx = max(pix)
	# print(max(pix))
	print(pix.index(mx))
	limit = pix.index(mx)

	# Threshold the image - Anything below intensity 80 gets set to white
	Ithresh = Ihmf2 < limit*0.8
	Ithresh = 255*Ithresh.astype("uint8")
	# Ithresh = cv2.copyMakeBorder(Ithresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, (0,0,0))


	# Clear off the border.	Choose a border radius of 15 pixels
	Iclear = imclearborder(Ithresh, 2)

	# Eliminate regions that have areas below 120 pixels
	Iopen = bwareaopen(Iclear, 0.003*(Iclear.shape[0]*Iclear.shape[1]))

	Iopen = cv2.copyMakeBorder(Iopen, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, (0,0,0))


	# Show all images
	# cv2_imshow(img)
	# cv2_imshow(Ihmf2)
	# cv2_imshow(Ithresh)
	# cv2_imshow(Iclear)
	# cv2_imshow(Iopen)
	return Iopen


def refine_answer(ans_lis):
	if(len(ans_lis)<4):
		return ans_lis
	if ans_lis[1] == 'H':
		ans_lis[0] = 'M'
	
	if ans_lis[1] == 'L' and (ans_lis[0] == '0' or ans_lis[0] == 'O'):
		ans_lis[0] = 'D'
	
	if ans_lis[1] == 'J':
		ans_lis[0] = 'G'
	
	
	if ans_lis[0] == 'G':
		ans_lis[1] = 'J'
	
	if ans_lis[1] == 'N':
		ans_lis[0] = 'T'
	
	
	if ans_lis[0] == 'T':
		ans_lis[1] = 'N'

	if ans_lis[1] == 'B' and (ans_lis[0]=='M' or ans_lis[0] == 'N'):
		ans_lis[0] = 'W'

	# ABCDEFGHIJKLMNOPQRSTUVWXYZ
	# excahnge = "48C0EF6H1JK4MN0P0R570VWXY2"

	if ans_lis[2] == 'I':
		ans_lis[2] = '1'
	if ans_lis[3] == 'I':
		ans_lis[3] = '1'
	
	
	if ans_lis[2] == 'L':
		ans_lis[2] = '4'
	if ans_lis[3] == 'L':
		ans_lis[3] = '4'
	
	
	if ans_lis[2] == 'Q':
		ans_lis[2] = '0'
	if ans_lis[3] == 'Q':
		ans_lis[3] = '0'

	
	if ans_lis[2] == 'Z':
		ans_lis[2] = '2'
	if ans_lis[3] == 'Z':
		ans_lis[3] = '2'
	


	for i in range(len(ans_lis)):
		if ans_lis[i] == 'O':
			ans_lis[i] = '0'

	check = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	if len(ans_lis) >= 8:
		if ans_lis[len(ans_lis) - 1] not in check:
			if ans_lis[len(ans_lis) - 1] == 'B':
				ans_lis[len(ans_lis) - 1] = '8'
			
			if ans_lis[len(ans_lis) - 1] == 'S':
				ans_lis[len(ans_lis) - 1] = '5'

			if ans_lis[len(ans_lis) - 1] == 'I':
				ans_lis[len(ans_lis) - 1] = '1'
			
			if ans_lis[len(ans_lis) - 1] == 'Q':
				ans_lis[len(ans_lis) - 1] = '0'

		if ans_lis[len(ans_lis) - 2] not in check:
			if ans_lis[len(ans_lis) - 2] == 'B':
				ans_lis[len(ans_lis) - 2] = '8'

			if ans_lis[len(ans_lis) - 2] == 'S':
				ans_lis[len(ans_lis) - 2] = '5'

			if ans_lis[len(ans_lis) - 2] == 'I':
				ans_lis[len(ans_lis) - 2] = '1'
			
			if ans_lis[len(ans_lis) - 2] == 'Q':
				ans_lis[len(ans_lis) - 2] = '0'

		if ans_lis[len(ans_lis) - 3] not in check:
			if ans_lis[len(ans_lis) - 3] == 'B':
				ans_lis[len(ans_lis) - 3] = '8'

			if ans_lis[len(ans_lis) - 3] == 'I':
				ans_lis[len(ans_lis) - 3] = '1'
			
			if ans_lis[len(ans_lis) - 3] == 'Q':
				ans_lis[len(ans_lis) - 3] = '0'
			if ans_lis[len(ans_lis) - 3] == 'S':
				ans_lis[len(ans_lis) - 3] = '5'
			

		if ans_lis[len(ans_lis) - 4] not in check:
			if ans_lis[len(ans_lis) - 4] == 'B':
				ans_lis[len(ans_lis) - 4] = '8'

			if ans_lis[len(ans_lis) - 4] == 'I':
				ans_lis[len(ans_lis) - 4] = '1'
			
			if ans_lis[len(ans_lis) - 4] == 'Q':
				ans_lis[len(ans_lis) - 4] = '0'
			
			if ans_lis[len(ans_lis) - 4] == 'S':
				ans_lis[len(ans_lis) - 4] = '5'

	return ans_lis


def Get_Segmented(Iopen):
	kernel = np.ones((3,3),np.uint8)
	Iopen = cv2.erode(Iopen,kernel,iterations = 1)
	ctrs, _ = cv2.findContours(Iopen.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
	img_area = Iopen.shape[0]*Iopen.shape[1]
	# print(Iopen.shape[0],Iopen.shape[1])
	letters = []
	for i, ctr in enumerate(sorted_ctrs):
		area = cv2.contourArea(ctr)
		x, y, w, h = cv2.boundingRect(ctr)
		mx= max(w,h)
		roi_area = w*h
		roi_ratio = roi_area/img_area
		if mx*mx >= Iopen.shape[0]*Iopen.shape[0]*0.03 and w>0.015 * Iopen.shape[1] and h>0.075 * Iopen.shape[0] and h <0.65 * Iopen.shape[0] and  w <0.25 * Iopen.shape[1]:
			# cv2.rectangle(Iopen,(x,y),( x + w, y + h ),(90,0,255),1)
			# print(w,h)
			letters.append([x,y,w,h])
	letters = refine_contours(letters)
	letters = sorted(letters, key=lambda ctr: ctr[0])
	segmented = []
	for coord in letters:
		[x, y, w, h] = coord
		# print("x, y, w, h = ",x, y, w, h)
		# print(image_result.shape[0],image_result.shape[1])
		seg_letter = Iopen[(int)(max(0, y - 0.2 * h)) : (int)(min(Iopen.shape[0], y + 1.2 * h)), (int)(max(0, x - 0.05 * w)) : (int)(min(Iopen.shape[1], x + 1.05 * w))]
		seg_letter = refine_final(seg_letter)
		# giving some black border to the images
		limit = max(5,(int)((h-w+5)/2))
		seg_letter = cv2.copyMakeBorder(seg_letter, 5, 5, limit, limit, cv2.BORDER_CONSTANT)
		segmented.append(seg_letter)
	return segmented

