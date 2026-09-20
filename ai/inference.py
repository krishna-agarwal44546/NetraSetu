import sys
import os
import gc
import json
import cv2
import numpy as np
import torch
import timm

from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download


# ============================================================
# CONFIG
# ============================================================

MODEL_REPO = "ClementP/FundusDRGrading-resnet18"

# Lowered from 512 -> 256 by default. This roughly quarters the
# memory used by every activation map in the network (memory
# scales with width x height), which matters a lot on a 512MB
# instance. Override with the INPUT_SIZE env var if you have
# more headroom (e.g. running locally or on a bigger instance).
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "256"))

# Absolute paths make the script work when Node.js launches it
# from a different working directory (important on Render).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Render's filesystem is ephemeral, so keep the Hugging Face cache
# outside the source tree. The model is downloaded only once per
# running service because we also keep the loaded model in memory.
HF_CACHE_DIR = os.getenv(
    "HF_CACHE_DIR",
    "/tmp/huggingface"
)

CLASSES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# Loaded once and reused for subsequent requests.
MODEL = None


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(
    f"Using device: {DEVICE}",
    file=sys.stderr
)

# Render will normally run on CPU. Keep CPU usage predictable.
try:
    torch.set_num_threads(
        int(os.getenv("TORCH_NUM_THREADS", "2"))
    )
except ValueError:
    pass


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (INPUT_SIZE, INPUT_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global MODEL

    # IMPORTANT for Render:
    # Do not download/load the model for every uploaded image.
    if MODEL is not None:
        return MODEL

    print(
        "Loading DR model...",
        file=sys.stderr
    )

    os.makedirs(
        HF_CACHE_DIR,
        exist_ok=True
    )

    # Optional Hugging Face token. Public repositories work without it,
    # but HF_TOKEN can be supplied as a Render environment variable.
    hf_token = os.getenv("HF_TOKEN")

    download_args = {
        "repo_id": MODEL_REPO,
        "filename": "model.safetensors",
        "cache_dir": HF_CACHE_DIR
    }

    if hf_token:
        download_args["token"] = hf_token

    print(
        "Downloading/checking DR model weights from Hugging Face...",
        file=sys.stderr,
        flush=True
    )

    weights_path = hf_hub_download(
        **download_args
    )

    print(
        f"Model weights available at: {weights_path}",
        file=sys.stderr,
        flush=True
    )

    # This checkpoint uses regression:
    # fc = Linear(2048, 1)
    #
    # The output is a continuous DR severity score from which
    # the prototype maps to grades 0-4 below.
    model = timm.create_model(
        "resnet18",
        pretrained=False,
        num_classes=1
    )

    from safetensors.torch import load_file

    print(
        "Loading safetensors weights...",
        file=sys.stderr,
        flush=True
    )

    state_dict = load_file(
        weights_path
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print(
        f"Moving model to {DEVICE}...",
        file=sys.stderr,
        flush=True
    )

    model = model.to(DEVICE)

    model.eval()

    MODEL = model

    print(
        "DR model loaded successfully.",
        file=sys.stderr,
        flush=True
    )

    return MODEL


# ============================================================
# IMAGE QUALITY CHECK
# ============================================================

def assess_image_quality(image_path):

    print(
        "Checking image quality...",
        file=sys.stderr
    )

    img = cv2.imread(image_path)

    if img is None:

        return {
            "usable": False,
            "score": 0,
            "issues": [
                "Unable to read image"
            ]
        }

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    height, width = img.shape[:2]

    issues = []

    if width < 300 or height < 300:
        issues.append("Image resolution is too low")

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------------
    # Blur detection (variance of Laplacian)
    # --------------------------------------------------------

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # --------------------------------------------------------
    # Brightness / Contrast / Saturation
    # --------------------------------------------------------

    brightness = np.mean(gray)
    contrast = np.std(gray)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])

    if brightness < 30:
        issues.append("Image is too dark")

    if brightness > 225:
        issues.append("Image is overexposed")

    if contrast < 20:
        issues.append("Image has very low contrast")

    if blur_score < 50:
        issues.append("Image appears blurry")

    # --------------------------------------------------------
    # Fundus field-of-view check
    # --------------------------------------------------------

    gray_resized = cv2.resize(gray, (256, 256))

    non_black = np.sum(gray_resized > 15)
    total_pixels = gray_resized.shape[0] * gray_resized.shape[1]
    field_ratio = non_black / total_pixels

    if field_ratio < 0.25:
        issues.append("Fundus field of view is too small")

    # --------------------------------------------------------
    # QUALITY SCORE
    # --------------------------------------------------------

    score = 100

    if blur_score < 50:
        score -= 30
    elif blur_score < 100:
        score -= 15

    if brightness < 30 or brightness > 225:
        score -= 20

    if contrast < 20:
        score -= 20
    elif contrast < 30:
        score -= 10

    if field_ratio < 0.25:
        score -= 25
    elif field_ratio < 0.40:
        score -= 10

    score = max(0, min(100, score))

    usable = (len(issues) == 0 and score >= 60)

    result = {
        "usable": usable,
        "score": round(float(score), 2),
        "metrics": {
            "blurScore": round(float(blur_score), 2),
            "brightness": round(float(brightness), 2),
            "contrast": round(float(contrast), 2),
            "saturation": round(float(saturation), 2),
            "fieldOfViewRatio": round(float(field_ratio), 3),
            "width": width,
            "height": height
        },
        "issues": issues
    }

    # Free the full-resolution image arrays now that we're done
    # with them, rather than waiting for the function to return.
    del img, gray, hsv, gray_resized

    return result


# ============================================================
# FUNDUS PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    image.close()

    return tensor


# ============================================================
# GRAD-CAM
# ============================================================
#
# generate() now does BOTH the prediction and the Grad-CAM
# heatmap in a single forward + backward pass, instead of running
# the model twice (once under no_grad for the score, once again
# with gradients enabled for the heatmap). This removes one full
# ResNet50 forward pass per request.

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self.forward_hook
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self.backward_hook
        )

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor):
        """
        Runs a single forward + backward pass.
        Returns (cam, score) so the caller doesn't need a
        separate no_grad() forward pass just to get the score.
        """

        self.model.zero_grad(set_to_none=True)

        output = self.model(tensor)

        # Regression output
        score_tensor = output[0, 0]

        # Capture the raw score value BEFORE backward(), since
        # backward() frees parts of the graph.
        score = score_tensor.item()

        # Backpropagate regression score
        score_tensor.backward()

        if self.gradients is None:
            raise RuntimeError("Grad-CAM gradients were not captured.")

        gradients = self.gradients
        activations = self.activations

        # Global average pooling
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted feature maps
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # Remove negative values
        cam = torch.relu(cam)

        # Resize to model input size
        cam = torch.nn.functional.interpolate(
            cam,
            size=(INPUT_SIZE, INPUT_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze()

        # Normalize
        cam -= cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam /= max_value

        cam_np = cam.detach().cpu().numpy()

        # Explicitly drop references to the large intermediate
        # tensors now that we've copied out what we need. This
        # matters on a tight memory budget more than it would
        # normally, since Python doesn't guarantee immediate
        # collection otherwise.
        del output, score_tensor, gradients, activations, weights, cam
        self.activations = None
        self.gradients = None

        return cam_np, score

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# SAVE GRAD-CAM
# ============================================================

def save_gradcam(image_path, cam, output_path):

    original = cv2.imread(image_path)
    original = cv2.resize(original, (INPUT_SIZE, INPUT_SIZE))

    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(output_path, overlay)

    del original, heatmap, overlay


# ============================================================
# DR ANALYSIS
# ============================================================

def analyze(image_path):

    if not os.path.exists(image_path):
        return {
            "status": "error",
            "message": "Image file does not exist."
        }

    # --------------------------------------------------------
    # QUALITY CHECK
    # --------------------------------------------------------

    quality = assess_image_quality(image_path)

    if not quality["usable"]:
        return {
            "status": "unusable",
            "message": "Fundus image quality is insufficient for reliable analysis.",
            "quality": quality,
            "prediction": None,
            "recommendation": "Please capture another fundus image with better focus, illumination and field of view."
        }

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    tensor = preprocess_image(image_path)

    # --------------------------------------------------------
    # PREDICTION + GRAD-CAM (single pass)
    # --------------------------------------------------------

    print("Running DR prediction and Grad-CAM...", file=sys.stderr)

    # Last convolutional layer of ResNet50
    target_layer = model.layer4[-1]

    gradcam = GradCAM(model, target_layer)

    cam, score = gradcam.generate(tensor)

    gradcam.close()

    print(f"Raw DR model score: {score:.4f}", file=sys.stderr, flush=True)

    # --------------------------------------------------------
    # Convert regression score to DR grade
    # --------------------------------------------------------

    score = max(0.0, min(4.0, score))

    predicted_index = int(round(score))
    predicted_index = max(0, min(4, predicted_index))

    predicted_class = CLASSES[predicted_index]

    # --------------------------------------------------------
    # SAVE HEATMAP
    # --------------------------------------------------------

    filename = (
        os.path.splitext(os.path.basename(image_path))[0]
        + "_gradcam.jpg"
    )

    heatmap_path = os.path.join(OUTPUT_DIR, filename)

    save_gradcam(image_path, cam, heatmap_path)

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    result = {
        "status": "success",
        "quality": quality,
        "prediction": {
            "class": predicted_class,
            "grade": predicted_index,
            "severityScore": round(float(score), 3)
        },
        "explainability": {
            "gradcam": heatmap_path,
            "gradcamFilename": filename,
            "description": "Highlighted regions show areas that contributed strongly to the model's DR severity prediction."
        },
        "recommendation": get_recommendation(predicted_index)
    }

    # --------------------------------------------------------
    # MEMORY CLEANUP
    # --------------------------------------------------------
    # Explicitly release the input tensor and heatmap array, then
    # force garbage collection right after the memory-heavy step
    # rather than waiting for it to happen on its own. This matters
    # more on a tight (e.g. 512MB) instance than it would with
    # plenty of headroom.

    del tensor, cam
    gc.collect()

    return result


# ============================================================
# RECOMMENDATION
# ============================================================

def get_recommendation(grade):

    if grade == 0:
        return (
            "No apparent diabetic retinopathy. "
            "Routine screening should still be maintained."
        )

    elif grade == 1:
        return (
            "Mild diabetic retinopathy detected. "
            "Consider ophthalmic evaluation and follow-up."
        )

    elif grade == 2:
        return (
            "Moderate diabetic retinopathy detected. "
            "Ophthalmic evaluation is recommended."
        )

    elif grade == 3:
        return (
            "Severe diabetic retinopathy suspected. "
            "Prompt ophthalmic evaluation is recommended."
        )

    else:
        return (
            "Proliferative diabetic retinopathy suspected. "
            "Urgent ophthalmic evaluation is recommended."
        )


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            json.dumps({
                "status": "error",
                "message": "Image path required."
            }),
            file=sys.stdout
        )

        sys.exit(1)

    image_path = sys.argv[1]

    try:
        result = analyze(image_path)

        print(json.dumps(result, indent=2))

    except Exception as e:

        print(
            json.dumps({
                "status": "error",
                "message": str(e)
            })
        )

        sys.exit(1)