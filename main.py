import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('./dataset/IMG_1570.MOV')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame
    cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

# img_ = cv2.imread('original_image_left.jpg')
# #img_ = cv2.resize(img_, (0,0), fx=1, fy=1)
# img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)

# img = cv2.imread('original_image_right.jpg')
# #img = cv2.resize(img, (0,0), fx=1, fy=1)
# img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# sift = cv2.xfeatures2d.SIFT_create()
# # find the key points and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# #cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_,kp1,None))


# #FLANN_INDEX_KDTREE = 0
# #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# #search_params = dict(checks = 50)
# #match = cv2.FlannBasedMatcher(index_params, search_params)
# match = cv2.BFMatcher()
# matches = match.knnMatch(des1,des2,k=2)


# good = []
# for m,n in matches:
#     if m.distance < 0.03*n.distance:
#         good.append(m)


# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    flags = 2)

# img3 = cv2.drawMatches(img_,kp1,img,kp2,good,None,**draw_params)
# cv2.imshow("original_image_drawMatches.jpg", img3)