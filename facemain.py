import cv2
from random import randrange

#pre-tained data on face frontals (haarcascade_frontal)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#colored image
# img = cv2.imread("RDJ.jpg")

#to use webcam 
webcam = cv2.VideoCapture(0)


#iterate forever frames 
while True:

    ####Read the current frame
    successful_frame_read, frame = webcam.read()

    #grayscaled (BGR)
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #frame or img



    #Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    #draw
    for(x, y, w, h) in face_coordinates:   #loop used to iterate over multiple faces in one image
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(150, 256), randrange(150, 256), randrange(250, 256)), 2)

    cv2.imshow('AI face detection', frame)
    key = cv2.waitKey(1)

    ##### Force quit
    if key==81 or key==113:
        break

    # ###release 
    # webcam.release()

print("Code Copleted !")
'''
#grayscaled (BGR)
grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #frame or img

#Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

print(face_coordinates)
'''

#draw the rectangle

# for(x, y, w, h) in face_coordinates:   #loop used to iterate over multiple faces in one image
#     cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(150, 256), randrange(150, 256), randrange(250, 256)), 2)

#titlebar
# cv2.imshow('AI face detection', img)
# cv2.waitKey()

