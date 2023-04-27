import cv2
import imutils
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import time
import concurrent.futures
from scipy.ndimage.filters import convolve
# from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.QtWidgets import QApplication, QWidget
start = time.time()


def prewitt_filter(img):
    """
    This function applies the Prewitt edge filter to the input image.
    """
    weight_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    fx_convolve = np.abs(convolve(img.astype("float64"), weight_x))
    weight_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    fy_convolve = np.abs(convolve(img.astype("float64"), weight_y))
    f3_convolve = np.sqrt(fx_convolve ** 2 + fy_convolve ** 2)
    return f3_convolve


def detect(frame, mode, detector, show=True, confidence=0.7, fps=None):
    """
    This function receives a detector (HOG from OpenCV) and an image to scan and it shows and returns info about the people scanned.
    The show option can be turned off just for analysis purposes and also the confidence filter can be set at will. If a colormap
    is sent via the mode parameter it is applied before scan.
    """
    if mode is None:
        pass
    else:
        frame = cv2.applyColorMap(frame, mode)

    cp1 = time.time()
    box_cordinates, weights = detector.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
    cp2 = time.time()
    # print("[INFO] Time ellapsed per detection:\t" + str(round(cp2 - cp1, 3)) + " s")
    person = 1
    weights = np.array(weights)
    no_people = weights <= confidence
    weights = np.delete(weights, no_people)

    box_cordinates = np.delete(box_cordinates, no_people, axis=0)

    if show:
        # show_people(frame, box_cordinates, fps)
        for i, (x, y, w, h) in enumerate(box_cordinates):
            rectangle_color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)
            text_color = (0, 0, 255)
            cv2.putText(frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            person += 1

        status_color = (255, 0, 0)
        cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f'Total People : {person-1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow('Analysing Video...', frame)
    else:
        for i, (x, y, w, h) in enumerate(box_cordinates):
            person += 1

    return frame, person, box_cordinates


def humanDetector(args):
    """
    This function receives information about what the user wants to do and decides how the detector is applied.
    """
    image_path = args.image
    video_path = args.video
    output_path = args.output
    mode = args.mode
    detector = args.detector
    show = args.show
    confidence = args.confidence
    prewitt = args.prewitt
    if str(args.camera) == 'True':
        camera = True
    else:
        camera = False
    writer = None
    if output_path is not None and image_path is None:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600))
    if camera:
        detectByCamera(output_path, writer, mode, detector, show=show, confidence=confidence, prewitt=prewitt)
    elif video_path is not None:
        frame, people, coords, fps = detectByPathVideo(video_path, writer, mode, detector, show=show, confidence=confidence, prewitt=prewitt)
    elif image_path is not None:
        frame, people, coords, fps = detectByPathImage(image_path, output_path, mode, detector, show=show, confidence=confidence, prewitt=prewitt)
    return frame, people, coords, fps


def detectByCamera(output_path, writer, mode, detector, show=True, confidence=0.7, prewitt=False):
    """
    This funciton receives some info relevamt for the detections and uses the computer camera as the source images. It doesn't return any information,
    only for show purposes.
    """
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print('[INFO] Detecting people...\t(Exit with \'q\')')
    while True:
        check, frame = video.read()
        if prewitt:
            channels = [prewitt_filter(frame[:, :, 0]), prewitt_filter(frame[:, :, 1]), prewitt_filter(frame[:, :, 2])]
            for i in range(3):
                frame[:, :, i] = channels[i]
        frame, people, coords = detect(frame, mode, detector, show=show, confidence=confidence)
        cv2.imshow('Camera', frame)
        if writer is not None:
            writer.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


def detectByPathVideo(path, writer, mode, detector, show=True, confidence=0.7, prewitt=False):
    """
    This funciton receives a path to a video and opens it frame by frame to apply the detector.
    """
    video = cv2.VideoCapture(path)
    frame_count = 0
    length_id = int(cv2.CAP_PROP_FRAME_COUNT)
    width_id = int(cv2.CAP_PROP_FRAME_WIDTH)
    height_id = int(cv2.CAP_PROP_FRAME_HEIGHT)
    length = int(cv2.VideoCapture.get(video, length_id))
    width = int(cv2.VideoCapture.get(video, width_id))
    height = int(cv2.VideoCapture.get(video, height_id))
    check, frame = video.read()
    cp1 = time.time()
    time.sleep(0.1)
    if not check:
        print('Video Not Found. Please Enter a Valid Path (Full Path from Working Directory Should be Provided).')
        return
    print('[INFO] Detecting people...\t(Exit with \'q\')')
    people_hist = np.array([])
    coords_hist = np.ones((4, 1))
    while video.isOpened():
        # check is True if reading was successful
        check, frame = video.read()
        if prewitt:
            channels = [prewitt_filter(frame[:, :, 0]), prewitt_filter(frame[:, :, 1]), prewitt_filter(frame[:, :, 2])]
            for i in range(3):
                frame[:, :, i] = channels[i]
        if check:
            frame_count += 1
            frame_percent = round((frame_count / length) * 100, 2)
            fps = round(1 / (time.time() - cp1), 3)
            cp1 = time.time()
            if frame_count % 10 == 0:
                print(f"[PROCESSING] {frame_percent}%")

            frame = imutils.resize(frame, width=min(800, frame.shape[1]))
            fps_color_g = (fps < 15) * (255 * fps / 15) + 255 * (fps >= 15)
            fps_color_r = (fps < 15) * (255 * (15 - fps) / 15) + 0 * (fps >= 15)
            fps_color = (fps_color_r, fps_color_g, 0)
            cv2.putText(frame, f'FPS : {fps}', (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, fps_color, 2)
            frame, people, coords = detect(frame, mode, detector, show=show, confidence=confidence)

            people_hist = np.append(people_hist, people)

            if coords.shape[0] != 0:
                coords_hist = np.append(coords_hist, coords[0].reshape((4, 1)), axis=1)

            if frame_count == 1:
                coords_hist = coords_hist[:, 1:]

            if writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                print("[PROCESSING] Terminating scan... ")
                break
        else:
            print("[PROCESSING] 100%")
            break

    video.release()
    cv2.destroyAllWindows()
    return frame, people_hist, coords_hist, fps


def detectByPathImage(path, output_path, mode, detector, show=True, confidence=0.7, prewitt=False):
    """
    This function receives a path to an image and opens it to apply the detector.
    """
    image = cv2.imread(path)
    image = imutils.resize(image, width=min(800, image.shape[1]))
    if prewitt:
        channels = [prewitt_filter(image[:, :, 0]), prewitt_filter(image[:, :, 1]), prewitt_filter(image[:, :, 2])]
        for i in range(3):
            image[:, :, i] = channels[i]
    cp1 = time.time()
    frame, people, coords = detect(image, mode, detector, show=show, confidence=confidence)
    cp2 = time.time()
    fps = round(1 / (cp2 - cp1), 3)
    if output_path is not None:
        cv2.imwrite(output_path, frame)
    cv2.waitKey(timer)
    cv2.destroyAllWindows()
    return frame, people, coords, fps


def get_colormap(colormaps, _map=None):
    if _map is None:
        print("[INFO] Using Colormap: Default (RGB)")
        return None
    elif _map not in list(colormaps.keys()):
        print("[WARNING] Invalid colormap chosen. Applying default RGB mode. Else try: ", list(colormaps.keys()))
        return None
    else:
        return colormaps[_map]


class parameters:
    """
    This class is used as a parameter box. It receives all the info needed at once and creates a box where the info can be retreieved from.
    """

    def __init__(self, detector, output_path=None, path2video=None, path2image=None, camera=False, mode=None, show=True, confidence=0.7, prewitt=False):
        self.output = output_path
        self.video = path2video
        self.image = path2image
        self.camera = camera
        self.mode = mode
        self.detector = detector
        self.show = show
        self.confidence = confidence
        self.prewitt = prewitt


def colormaps():
    """
    This function is a tool to get all OpenCV colormaps available easily. It return the name and code together for each colormap for easy future use.
    """
    flags = [cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE, cv2.COLORMAP_CIVIDIS, cv2.COLORMAP_COOL, cv2.COLORMAP_DEEPGREEN, cv2.COLORMAP_HOT, cv2.COLORMAP_HSV, cv2.COLORMAP_INFERNO, cv2.COLORMAP_JET, cv2.COLORMAP_MAGMA, cv2.COLORMAP_OCEAN, cv2.COLORMAP_PARULA, cv2.COLORMAP_PINK, cv2.COLORMAP_PLASMA, cv2.COLORMAP_RAINBOW, cv2.COLORMAP_SPRING, cv2.COLORMAP_SUMMER, cv2.COLORMAP_TURBO, cv2.COLORMAP_TWILIGHT, cv2.COLORMAP_TWILIGHT_SHIFTED, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_WINTER]
    flags_name = [i for i in dir(cv2) if i.startswith('COLORMAP')]
    colormaps = [i[9:] for i in flags_name]
    colormaps = dict(zip(colormaps, flags))
    return colormaps


def compare(args):
    """
    [DEV function] This function creates threads of processing and queues them into the processor cores (CPU limited func). In each thread an image with a colormap applied
    is scanned and lastly all the info from the threads are put together. For a 4 core CPU it saves around 25%-30% of time for a 22 colormaps run. It also shows performance between threads.
    NOT INTENDED FOR SHOWING OFF
    [FUNCTION POTENTIAL WARNING] This function could be used to thread a video scanning and enhance the frames analysed per second by a 25% approx.
    """
    image_path = args.image
    video_path = args.video
    detector = args.detector
    show = args.show
    output_path = args.output[:-4]
    confidence = args.confidence

    cp1 = time.time()
    filetype(args)
    if video_path is not None:
        params = parameters(detector, output_path=output_path + "_default.mp4", path2video=video_path, mode=None, show=show)
        frame, people_d, coords_d, fps = humanDetector(params)
    elif image_path is not None:
        params = parameters(detector, output_path=output_path + "_default.jpg", path2image=image_path, mode=None, show=show)
        frame, people_d, coords_d, fps = humanDetector(params)
    else:
        print("[Warning] Enter a valid path to a video or image (Don't use absolute paths).")
    cp2 = time.time()
    print("\n[INFO] Time ellapsed for default colormap (RGB):\t" + str(round(cp2 - cp1, 3)) + "s")
    coords = np.array([coords_d])
    people_d = np.array([people_d])
    labels = np.array(["DEFAULT"])
    counter = 1
    colormap = colormaps()
    n_colormaps = len(list(colormap.keys()))
    for i, v in enumerate(list(colormap.keys())):
        if i % 4 == 0:
            counter += 1
            labels = np.append(labels, v)
            _map = get_colormap(colormap, _map=v.upper())

            if video_path is not None:
                param = parameters(detector, output_path=output_path + "_" + str(v) + ".mp4", path2video=video_path, mode=_map, show=show, confidence=confidence)
            elif image_path is not None:
                param = parameters(detector, output_path=output_path + "_" + str(v) + ".jpg", path2image=image_path, mode=_map, show=show, confidence=confidence)

            params = np.append(params, param)
        else:
            continue

    color_str = ""
    for i in labels:
        color_str += i + ", "
    print("[INFO] Using", counter, "colormap(s)", "(" + str(round((counter / n_colormaps) * 100, 2)) + "%)")
    print("[INFO] Using Colormaps:", color_str[:-2])

    result = np.array([])
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(humanDetector, param) for param in params]
        result = np.array([f.result() for f in futures], dtype=object)

    frames = result[:, 0]
    people = result[:, 1]
    coords = result[:, 2]
    fps = result[:, 3]
    fps = round(fps.sum() / len(fps), 3)
    coords = np.array([list(coord) for coord in coords], dtype=object)  # coords[colormaps,people,[rectangle, weight]]
    cp6 = time.time()
    print("[INFO] Time ellapsed per frame:\t" + str(round(cp6 - cp1, 3)) + "s\n")
    values = people.astype("uint8")
    fig, ax = plt.subplots()
    bars = plt.bar(np.arange(len(people)), values, 0.8)
    ax.set_xticks(np.arange(len(people)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of People Detected")
    for p in bars:
        height = round(p.get_height(), 3)
        ax.annotate('{}'.format(height), xy=(p.get_x() + p.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    plt.yticks(color='w')
    fig.tight_layout()
    plt.show()
    return frames, people, coords, fps


def show_people(frame, coords, fps):
    """
    This function receives some rectangle coords and an image and presents them together.
    """
    cv2.destroyAllWindows()
    coords = np.array(coords, dtype=object)
    person = 1
    for i, (x, y, w, h) in enumerate(coords[:, 0]):
        rectangle_color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)
        text_color = (0, 0, 255)
        cv2.putText(frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        person += 1

    status_color = (255, 0, 0)
    cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)
    fps_color = (255, 0, 0)
    cv2.putText(frame, f'FPS : {fps}', (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, fps_color, 2)
    cv2.putText(frame, f'Total People : {person-1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('Analysing Video...', frame)
    cv2.waitKey(timer)
    cv2.destroyAllWindows()
    return


def filetype(args):
    """
    This function receives the paramaters box and retrieves the used file name for info purposes.
    """
    if args.video is not None:
        print('[INFO] Opening Video from path.\n')
        path = args.video[::-1]
        filename = path[path.index("\\") - 1::-1]
        print("[INFO] Using file: ", filename)
    elif args.image is not None:
        print('[INFO] Opening Image from path.\n')
        path = args.image[::-1]
        filename = path[path.index("\\") - 1::-1]
        print("[INFO] Using file:", filename)
    elif args.camera is not None:
        print('[INFO] Opening Web Cam.\n')
    return


def test(detector, show=True, confidence=0.7):
    """
    [DEV FUNCTION] This function prepares the parameter box for the compare threaded test. NOT INTENDED FOR SHOWING OFF
    """
    output_path = wd + "\\processed\\test_image1_checked.jpg"
    image_path = wd + "\\dataset\\test_image1.jpg"
    param = parameters(detector, output_path=output_path, path2image=image_path, show=show, confidence=confidence)
    frames, people_score, coords, fps = compare(param)
    return frames, people_score, coords, fps


def default_image(detector, show=True, confidence=0.7, prewitt=False, _map=None):
    """
    This function is designed to ease the normal use of the human detector. The user sets up the parameters box with the variables below and the programm runs an ordinary scan.
    """
    output_path = wd + "\\processed\\test_image1_checked.jpg"
    image_path = wd + "\\dataset\\test_image1.jpg"

    _map = _map
    if _map is not None:
        _map = _map.upper()
    colormap = get_colormap(colormaps(), _map=_map)
    param = parameters(detector, output_path=output_path, path2image=image_path, mode=colormap, show=show, confidence=confidence, prewitt=prewitt)

    frames, people, coords, fps = humanDetector(param)
    return frames, people, coords, fps


def default_video(detector, show=True, confidence=0.7, prewitt=False, _map=None):
    """
    This function is designed to ease the normal use of the human detector. The user sets up the parameters box with the variables below and the programm runs an ordinary scan.
    """
    # output_path = wd + "\\processed\\test1_10s_cut_checked.mp4"
    video_path = wd + "\\dataset\\test1_10s_cut.mp4"

    _map = _map
    if _map is not None:
        _map = _map.upper()
    colormap = get_colormap(colormaps(), _map=_map)
    param = parameters(detector, path2video=video_path, mode=colormap, show=show, confidence=confidence, prewitt=prewitt)  # Insert 'output_path=output_path' parameter to save the scanned video file (not working due to an unknown error, it saves the file but cannot be opened correctly)

    frames, people, coords, fps = humanDetector(param)
    return frames, people, coords, fps


def box_coordinates_analysis(coords, normalization):
    normalization = normalization / 1080
    coords = np.sort(coords, axis=0)
    shape = coords.shape[0]
    vgp = np.ones((shape, 2), dtype=object)
    surface = np.ones((shape, 1), dtype=object)
    CM = np.ones((shape, 2), dtype=object)
    ratio = np.ones((shape, 1), dtype=object)
    CM_vgp = np.ones((shape, 1), dtype=object)
    score = np.ones((shape, 1), dtype=object)
    for i, (x, y, w, h) in enumerate(coords):
        vgp[i] = np.array([x + w / 2, y + h]) / normalization
        surface[i] = w * h / normalization
        CM[i] = np.array([x + w / 2, y + h / 2]) / normalization
        ratio[i] = y / (x * normalization)
        CM_vgp[i] = h / (2 * normalization)

    for i in range(shape):
        score[i] = (surface[i] * ratio[i] * CM_vgp[i])
    data = list(zip(vgp, surface, CM, ratio, CM_vgp))

    return score, data


def showcase():
    # Normal usage for image scanning with default, prewitt filter and colormap applied.
    frame, people, coords, fps = default_image(HOGCV, show=True, confidence=0.7)
    frame_p, people_p, coords_p, fps_p = default_image(HOGCV, show=True, confidence=0.7, prewitt=True)
    frame_c, people_c, coords_c, fps_c = default_image(HOGCV, show=True, confidence=0.7, _map="hsv")
    image_size = frame.shape[0] * frame.shape[1]
    score, data = box_coordinates_analysis(coords, image_size)
    c_data = np.ones((len(data)), dtype=object)

    print("\n[SHOWCASE] People Score (Not percentage. Higher values means less chance of a fall.)")
    for i, v in enumerate(score):
        print("Person nº" + str(i + 1) + ": " + str(round(v[0], 5)))
    print("[SHOWCASE] The above score is used to determine whether a detection is a fall or not. Time-dependent variables are nopt implemented (velocity, acceleration, etc) thus this calculation is not precise enough for comercialization. Being able to implememnt time-dependent variables uncovers too many probles to handle as of now, it may require several days to fix them.\n")

    print("[SHOWCASE] Showing table with diferent variables extracted from image scanning\n")
    fig, ax = plt.subplots(1, 1)
    labels = ["Person ID", "Virtual Ground Point", "Surface", "Center of Mass", "y/x Ratio", "CM-VGP"]
    ax.axis('tight')
    ax.axis('off')
    ax.set_title("(Values are normalized to image size)")

    for i, v in enumerate(data):
        person = [str(i),
                  "(" + str(round(v[0][0], 5)) + ", " + str(round(v[0][1], 5)) + ")",
                  str(round(v[1][0], 5)),
                  "(" + str(round(v[2][0], 5)) + ", " + str(round(v[2][1], 5)) + ")",
                  str(round(v[3][0], 5)),
                  str(round(v[4][0], 5))]
        c_data[i] = person

    table = ax.table(cellText=c_data, colLabels=labels, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    plt.show()

    # Normal usage for video scanning.
    frame, people, coords, fps = default_video(HOGCV, show=True, confidence=0.7)

    print("\n[SHOWCASE] This programm can also use the computer camera for detection. Generally these types of cameras are bad positioned for our purpose beacuse they are not intended for people/personal monitoring. Thus the use of the camera for scanning is not showcased. Nevertheless uncomment the code below in case one may want to see it working (may not work anyway because of computer id assignment to webcams).")

    """
    # Normal usage for camera scanning.
    param = parameters(HOGCV, mode=None, show=True, confidence=0.7, camera=True)
    humanDetector(param)
    """

    return


if __name__ == "__main__":
    global timer, HOGCV
    timer = 5000  # Time in ms between images when compared. Set to 0 for unlimited.

    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    wd = str(pathlib.Path().absolute())
    print("[INFO] Current working directory:", "\n" + wd)

    print("[INFO] Available colormaps and whose OpenCV code:\n", colormaps())

    # frames, people, coords, fps = test(HOGCV, show=False, confidence=0.7)  # DEV option, not intended for showcase
    showcase()

    end = time.time()
    print("\nTime ellapsed:\t" + str(round(end - start, 3)) + " s")
