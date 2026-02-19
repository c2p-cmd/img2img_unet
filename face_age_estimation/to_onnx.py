import torch
from torchvision import transforms
from model import FaceCNN
import cv2

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(device)

model = FaceCNN()
model.load_state_dict(torch.load("face_model.pth", map_location=torch.device(device)))

model.eval()

x = cv2.imread("../UTKFace/29_1_0_20170104022706451.jpg.chip.jpg")
assert x is not None, "No image"
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_NEAREST)

t = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

x = t(x)

x = torch.reshape(x, (1, *x.shape))

print(x.shape)

print("Trying pass")

with torch.no_grad():
    o = model(x)

print("Done", o)

program = torch.onnx.export(
    model,
    (x, ),
    dynamo=True,
)

program.save("face_estimator.onnx")
print("Exported!")

