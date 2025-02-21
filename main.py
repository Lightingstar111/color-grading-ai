from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO

# Load AI Model (Make sure color_enhancer.pth is uploaded)
model = torch.jit.load("color_enhancer.pth")
model.eval()

app = FastAPI()

def enhance_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        enhanced_tensor = model(image_tensor)
    enhanced_image = transforms.ToPILImage()(enhanced_tensor.squeeze(0))
    output_buffer = BytesIO()
    enhanced_image.save(output_buffer, format="JPEG")
    return output_buffer.getvalue()

@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    image_bytes = await file.read()
    enhanced_image_bytes = enhance_image(image_bytes)
    return {"enhanced_image": enhanced_image_bytes}

@app.get("/")
def home():
    return {"message": "AI Color Grading API is running!"}
