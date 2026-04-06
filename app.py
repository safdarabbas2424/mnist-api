from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import io

app = FastAPI()

# -------------------------
# 1. Define same model class
# -------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

# -------------------------
# 2. Load saved model
# -------------------------
model = SimpleNN()
model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
model.eval()

# -------------------------
# 3. Preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -------------------------
# 4. Home route
# -------------------------
@app.get("/")
def home():
    return {"message": "MNIST API is running"}

# -------------------------
# 5. Prediction route
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    image = transform(image)
    image = image.unsqueeze(0)  # shape: [1, 1, 28, 28]

    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

    return {"predicted_digit": predicted_class}