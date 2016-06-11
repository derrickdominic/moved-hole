import cv2
import numpy as np
import matplotlib.pyplot as plt

# From http://docs.opencv.org/3.1.0/d1/de0/tutorial_py_feature_homography.html

MIN_MATCH_COUNT = 10

img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# # Debug by writing keypoint images
# gray1_kp1 = cv2.drawKeypoints(gray1, kp1, None)
# gray2_kp2 = cv2.drawKeypoints(gray2, kp2, None)
# cv2.imwrite('gray1_kp1.jpg', gray1_kp1)
# cv2.imwrite('gray2_kp2.jpg', gray2_kp2)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w,d = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
    exit()

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

#img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
warped1 = cv2.warpPerspective(img1, M, (img1.shape[1], img1.shape[0]))
cv2.imwrite('warped1.jpg', warped1)
