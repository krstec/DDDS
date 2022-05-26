
from tkinter import*
from tkinter import messagebox
from turtle import title
from PIL import ImageTk
import dlib
#from drowsiness_detector import Drowsiness_ 

class Login:
    def __init__(self,root):
        self.root=root
        self.root.title("Drowsiness Detection System")
        
        self.root.geometry("1199x600+100+50")
        self.bg= ImageTk.PhotoImage(file="images/ddds.jpg")
        
        self.bg_image=Label(self.root,image=self.bg).place(x=0,y=0,relwidth=1,relheight=1)
        title1 = Label(text="Drowsiness Detection System", font=("Imapct",35,"bold"),fg="#d77337",).place(x=90,y=30)        
        
        self.root.resizable(False,False),
        Frame_login = Frame(self.root,bg="white")
        Frame_login.place(x=150,y=150,height=340,width=500)



        title = Label(Frame_login, text="Login Here", font=("Imapct",35,"bold"),fg="#d77337",bg="white").place(x=90,y=30)
        desc = Label(Frame_login, text="Account Employee Login Area", font=("Goudy old style",15,"bold"),fg="#d25d17",bg="white").place(x=90,y=100)
        
        
        
        lbl_user = Label(Frame_login, text="Usernmae", font=("Goudy old style",15,"bold"),fg="gray",bg="white").place(x=90,y=140)
        self.text_user=Entry(Frame_login,font=("times new roman",15),bg="lightgray")
        self.text_user.place(x=90,y=170,width=350,height=35)


        lbl_pass = Label(Frame_login, text="Password", font=("Goudy old style",15,"bold"),fg="gray",bg="white").place(x=90,y=210)
        self.text_pass=Entry(Frame_login,font=("times new roman",15),bg="lightgray")
        self.text_pass.place(x=90,y=240,width=350,height=35)


        forget_btn = Button(Frame_login,text="Forget Password?",bg="white",fg="#d77337",bd=0,font=("times new roman",12)).place(x=90,y=280)
        forget_SignUp = Button(Frame_login,text="Create Account!",bg="white",fg="#d77337",bd=0,font=("times new roman",12)).place(x=330,y=280)
        login_btn = Button(self.root,command=self.login_function,text="Login",fg="white",bg="#d77337",font=("times new roman",20)).place(x=300,y=470,width=100,height=40)


    def login_function(self):
        if self.text_user.get()=="" or self.text_pass.get()=="":
            messagebox.showerror("error","All field are required",parent= self.root)
        elif self.text_user.get()!="dds" or self.text_pass.get()!="123456":
            messagebox.showerror("error","Invalid Username/password",parent=self.root)
        else :
            messagebox.showinfo("Welcome",f"Welcome{self.text_user.get()}",parent=self.root)
            root.destroy()  

            #predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            '''This script detects if a person is drowsy or not,using dlib and eye aspect ratio
            calculations. Uses webcam video feed as input.'''

            #Import necessary libraries
            from scipy.spatial import distance
            from imutils import face_utils
            import numpy as np
            import pygame #For playing sound
            import time
            import dlib 
            import cv2

            #Initialize Pygame and load music
            pygame.mixer.init()
            pygame.mixer.music.load('audio/alert.wav')

            #Minimum threshold of eye aspect ratio below which alarm is triggerd
            EYE_ASPECT_RATIO_THRESHOLD = 0.3

            #Minimum consecutive frames for which eye ratio is below threshold for alarm to be triggered
            EYE_ASPECT_RATIO_CONSEC_FRAMES = 50

            #COunts no. of consecutuve frames below threshold value
            COUNTER = 0

            #Load face cascade which will be used to draw a rectangle around detected faces.
            face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

            #This function calculates and return eye aspect ratio
            def eye_aspect_ratio(eye):
                A = distance.euclidean(eye[1], eye[5])
                B = distance.euclidean(eye[2], eye[4])
                C = distance.euclidean(eye[0], eye[3])

                ear = (A+B) / (2*C)
                return ear

            #Load face detector and predictor, uses dlib shape predictor file
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

            #Extract indexes of facial landmarks for the left and right eye
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

            #Start webcam video capture
            video_capture = cv2.VideoCapture(0)

            #Give some time for camera to initialize(not required)
            time.sleep(2)

            while(True):
                #Read each frame and flip it, and convert to grayscale
                ret, frame = video_capture.read()
                frame = cv2.flip(frame,1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                #Detect facial points through detector function
                faces = detector(gray, 0)

                #Detect faces through haarcascade_frontalface_default.xml
                face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

                #Draw rectangle around each face detected
                for (x,y,w,h) in face_rectangle:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                #Detect facial points
                for face in faces:

                    shape = predictor(gray, face)
                    shape = face_utils.shape_to_np(shape)

                    #Get array of coordinates of leftEye and rightEye
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]

                    #Calculate aspect ratio of both eyes
                    leftEyeAspectRatio = eye_aspect_ratio(leftEye)
                    rightEyeAspectRatio = eye_aspect_ratio(rightEye)

                    eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

                    #Use hull to remove convex contour discrepencies and draw eye shape around eyes
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    #Detect if eye aspect ratio is less than threshold
                    if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
                        COUNTER += 1
                        #If no. of frames is greater than threshold frames,
                        if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                            pygame.mixer.music.play(-1)
                            cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
                    else:
                        pygame.mixer.music.stop()
                        COUNTER = 0

                #Show video feed
                cv2.imshow('Video', frame)
                if(cv2.waitKey(1) & 0xFF == ord('q')):
                    break

            #Finally when video capture is over, release the video capture and destroyAllWindows
            video_capture.release()
            cv2.destroyAllWindows()

                        



#title1 = Label(text="Drowsiness Detection System", font=("Imapct",35,"bold"),fg="#d77337",bg="white").place(x=90,y=30)        
root=Tk()
obj = Login(root)
root.mainloop(),
