import cv2
import time
import os

def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    temp = 0
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        if True :
            rows, cols, channels = frame.shape
            frame = cv2.resize(frame, (cols/2, rows/2), interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_loc + "/%#05d.jpg" % (temp+1), frame)

            temp += 1
        count = count + 1
        # If there are no more frames left
        # if (count > (video_length-1)):
        if count > 100:
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break

input_loc = './video/banana.m4v'
output_loc = './frame/'
video_to_frames(input_loc, output_loc)


