import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_name = input("Enter the video name (without ext) ")
    video_path = "sample_videos/" + video_name + ".avi"
    out = cv2.VideoWriter(video_path, fourcc, 30, (640, 480))
    input("Enter to start recording and then press \'q\' to finish recording")

    cv2.namedWindow("Camera")
    while(cap.isOpened()):
        ret, frame = cap.read()

        out.write(frame)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()