# import cv2
# # vidcap = cv2.VideoCapture('./Data/Video/Test.mp4')
# success,image = vidcap.read()
# count = 0
# success = True
# while success:
#   success,image = vidcap.read()
#   cv2.imwrite("./Data/Images/frame%d.jpg" % count, image)     # save frame as JPEG file
#   if cv2.waitKey(10) == 27:                     # exit if Escape is hit
#       break
#   count += 1


import cv2
outDir = "./Data/Images/frame%d.jpg"
videoFile ='./Data/Video/Test.mp4'
vidcap = cv2.VideoCapture('./Data/Video/Test.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("./Data/Images/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read and Save frame %d: '% count, success)
  count += 1