##### Import libraries

from fastapi import FastAPI, File, UploadFile, HTTPException # type: ignore
from fastapi.responses import JSONResponse # type: ignore
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from models.model_file import ResNet18_FC
from fastapi.middleware.cors import CORSMiddleware
import io
#import imghdr  standard_imghdr-3.13.0.dist-info
import logging
import imageio



logging.basicConfig(level=logging.DEBUG)  # Or use INFO if you don't want verbose output
logger = logging.getLogger(__name__)




# Define the FastAPI app
app = FastAPI(max_upload_size=10 * 1024 * 1024)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your frontend domain
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load your trained model from the models directory
# Initialize the model
model = ResNet18_FC()

# Load the checkpoint (adjust the path as needed)
checkpoint = torch.load('models/trained_apgd_resnet18.pth', map_location=torch.device('cpu'))

# Load the model weights
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()


# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Define image transformation with normalization
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize with CIFAR-10 means and stds
])



def open_image(file_contents):

    try:

        # Use imageio to determine the image format by reading the content
        image = imageio.imread(io.BytesIO(file_contents))

        # Use Pillow to get the image format
        image_pil = Image.open(io.BytesIO(file_contents))
        image_format = image_pil.format  # This will give you the format as a string like 'JPEG', 'PNG', etc.

        logger.debug(f"Image format: {image_format}")
       

        # Convert the image to RGB if it's not already in that format
        return Image.fromarray(image).convert("RGB")

    except Exception as e:
         raise ValueError(f"Error opening image: {str(e)}")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    logger.debug("==== ENDPOINT CALLED ====") 

    logger.debug("Received file: %s", file.filename) # To check if file is received

    logger.debug("Debugging log message 1")


    if not file:
        return JSONResponse(content={"error": "No file uploaded"}, status_code=400)
    

    # Read the file content
    contents = await file.read()
    logger.debug(f"File contents read. Length: {len(contents)} bytes")  # Print the length of the file contents

    try:
        ### open image based on format
        image = open_image(contents)
    except Exception as e:
        raise HTTPException(status_code=415, detail=str(e))

    # Apply transformations (resize, to tensor, and normalize)
    # Add batch dimension (1, 3, 32, 32)
    try:
        image = transform(image).unsqueeze(0)  
    except Exception as e:
        raise HTTPException(status_code = 400, detail = f"Error applying image transformations: {str(e)}")
 

    # Make a prediction using the model
    logger.debug("Image transformed! Heading towards prediction!")
    try:
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(image)
            # Convert logits to probabilities using softmax
            probabilities = F.softmax(outputs, dim=1)

            # Get the predicted class (index of max probability)
            predicted_class_idx = torch.argmax(probabilities, 1)

            # Get the confidence (probability of the predicted class)
            confidence = probabilities[0, predicted_class_idx.item()].item()

            # Get the class name
            predicted_class = class_names[predicted_class_idx.item()]


        # Return the predicted class as a JSON response
        return JSONResponse(content={"prediction": predicted_class, "Confidence": round(confidence*100, 3)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
