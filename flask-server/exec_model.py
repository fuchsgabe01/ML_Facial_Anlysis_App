import torch
from torch import nn
import torchvision.transforms as T
import cv2
import numpy as np


def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(filter_path)
    bboxes = face_cascade.detectMultiScale(image, 1.3, 5)
    return bboxes


def draw_bounding_boxes(image, bboxes):
    for box in bboxes:
        x1, y1, w, h = box
        cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (0, 255, 0), 10)


def decrease_brightness(image, p1, p2, brightness_factor):

    mask = np.ones_like(image)
    mask[p1[1]:p2[1], p1[0]:p2[0]] = 0

    processed_image = image * (1 - mask * brightness_factor)

    return processed_image.astype(np.uint8)


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=7, n_channel=32, drop=0.2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(n_input, n_channel, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(drop)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(n_channel, 2*n_channel, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(drop)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(2*n_channel, 2*n_channel, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(drop)
        )
        self.layer4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 128)
        )
        self.layer5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, n_output)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return nn.functional.log_softmax(x, dim=1)


conversion = {0: 'angry', 1: 'disgust', 2: 'fear',
              3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

PATH = "model2.pt"

filter_path = "haarcascade_frontalface_default.xml"

device = torch.device('cpu')

model = M5()
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

trans = T.Compose([
    T.ToPILImage(),
    T.Grayscale(1),
    T.Resize((48, 48)),
    T.ToTensor()
])


# read image as numpy array
# image = cv2.imread(
#    "C:\\Users\\fuchsga\\Desktop\\WebApps\\FacialAnalysisWebApp\\me\\excited.jpg")


def compute_expression(image):
    im_raw = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # get bounding box
    bboxes = detect_faces(image)

    # draw bounding box
    draw_bounding_boxes(image, bboxes)

    x1, y1, w, h = bboxes[0]
    image = decrease_brightness(image, (x1, y1), (x1+w, y1+h), 0.8)

    # assuming only one bounding box
    x1, y1, w, h = bboxes[0]
    im1 = im_raw[y1:y1+h, x1:x1+w]

    resized_img = trans(im1)
    res = model(resized_img[None, :])
    softmax = np.exp(res.detach().numpy()[0])
    return conversion[res.argmax(dim=-1)[0].item()], softmax, image
