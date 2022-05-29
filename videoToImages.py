import cv2
outDir = "./Data/Images/%d.jpg"

videoFile = input('Enter Video File Name (file akan di write ke directory ./Data/Images): ')
# videoFile ='./Data/Video/Data.mp4'
vidcap = cv2.VideoCapture(videoFile)
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(outDir % count, image)     # save frame sebagai JPEG file      
  success,image = vidcap.read() # kalau fail berarti bisa masalah file/ tidak ada frame lagi
  print('Read and Save frame %d: '% count, success) 
  count += 1