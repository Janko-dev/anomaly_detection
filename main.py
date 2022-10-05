import numpy as np
import cv2 as cv


def main():
    cap = cv.VideoCapture(1)
    back_sub = cv.createBackgroundSubtractorKNN(500, 1000, True)
    # base_ret, base_frame = cap.read()
    # base_mask = back_sub.apply(base_frame)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        fg_mask = back_sub.apply(frame)
        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fg_mask)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
