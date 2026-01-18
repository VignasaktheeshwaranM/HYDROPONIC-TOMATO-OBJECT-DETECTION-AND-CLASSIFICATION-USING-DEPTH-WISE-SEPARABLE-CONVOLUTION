!pip install --no-deps ultralytics supervision

import os
import torch
import torch.nn as nn
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Config
# =========================
DATA_YAML = "/kaggle/input/hydroponics-tomato/data.yaml"
NUM_EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
BASE_MODEL = "yolov8n.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
FINAL_WEIGHTS = "/kaggle/working/yolov8n_ds_final.pt"
RESUME = False

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------
# Depthwise-Separable Conv
# -------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        in_ch, out_ch = conv.in_channels, conv.out_channels
        kernel_size, stride, padding, dilation = conv.kernel_size, conv.stride, conv.padding, conv.dilation
        bias = conv.bias is not None

        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(
            in_ch, out_ch, 1, stride=1, padding=0, bias=bias)

        # Optionally attempt to copy weights
        try:
            with torch.no_grad():
                w = conv.weight.data
                dw = w.mean(dim=0, keepdim=False)
                self.depthwise.weight.copy_(dw.unsqueeze(1))
                pw = w.view(out_ch, in_ch, -1).mean(dim=2)
                self.pointwise.weight.copy_(pw.unsqueeze(-1).unsqueeze(-1))
                if bias and conv.bias is not None:
                    self.pointwise.bias.copy_(conv.bias.data.clone())
        except Exception:
            pass

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

def should_replace(conv: nn.Conv2d) -> bool:
    ks = conv.kernel_size
    if isinstance(ks, tuple):
        if ks[0] == 1 and ks[1] == 1:
            return False
    elif ks == 1:
        return False
    if conv.groups != 1:
        return False
    if conv.in_channels <= 1 or conv.out_channels <= 1:
        return False
    return True

def replace_convs(module: nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and should_replace(child):
            setattr(module, name, DepthwiseSeparableConv(child))
            replaced += 1
        elif hasattr(child, "conv") and isinstance(child.conv, nn.Conv2d) and should_replace(child.conv):
            child.conv = DepthwiseSeparableConv(child.conv)
            replaced += 1
        else:
            replaced += replace_convs(child)
    return replaced

def get_nn_module(yolo_model):
    m = getattr(yolo_model, "model", None)
    if isinstance(m, nn.Module):
        return m
    inner = getattr(m, "model", None)
    if isinstance(inner, nn.Module):
        return inner
    raise RuntimeError("Could not locate nn.Module inside YOLO model")

# -------------------------
# IoU Calculation Functions
# -------------------------
def calculate_mean_iou(model, data_yaml, batch_size, img_size, device):
    results = model.val(
        data=data_yaml,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        plots=False,
        split='test'
    )
    return results.box.map50

def main():
    print(f"[info] device={DEVICE}, loading base model {BASE_MODEL}")
    base_model = YOLO(BASE_MODEL)
    pytorch_model_base = get_nn_module(base_model)

    print("[info] Calculating mAP50 (approx IoU) before depthwise-separable replacement...")
    iou_before = calculate_mean_iou(base_model, DATA_YAML, BATCH_SIZE, IMG_SIZE, DEVICE)
    print(f"Mean IoU before replacement (mAP50): {iou_before:.4f}")

    depthwise_pytorch_model = get_nn_module(base_model)
    replaced = replace_convs(depthwise_pytorch_model)
    print(f"[info] replaced {replaced} Conv2d -> DepthwiseSeparableConv")

    print("\n[info] Model Summary (After Depthwise-Separable Replacement):")
    base_model.info(verbose=True)

    base_model.model = depthwise_pytorch_model

    print("[info] Training depthwise-separable conv model...")
    results = base_model.train(
        data=DATA_YAML,
        epochs=NUM_EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=4,
        resume=RESUME,
        cache=False
    )

    ckpt_path = os.path.join(CHECKPOINT_DIR, "last_ckpt.pt")
    torch.save({"epoch": NUM_EPOCHS - 1, "model": depthwise_pytorch_model.state_dict()}, ckpt_path)
    torch.save(depthwise_pytorch_model.state_dict(), FINAL_WEIGHTS)
    print(f"[info] Training done. Final weights saved to {FINAL_WEIGHTS}")

    print("[info] Calculating mAP50 (approx IoU) after depthwise-separable replacement...")
    iou_after = calculate_mean_iou(base_model, DATA_YAML, BATCH_SIZE, IMG_SIZE, DEVICE)
    print(f"Mean IoU after replacement (mAP50): {iou_after:.4f}")

    print(f"[info] Running validation and generating confusion matrix")
    val_results = base_model.val(
        data=DATA_YAML,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        conf=0.25,
        split='test',
        plots=True
    )

    print(f"mAP50-95: {val_results.box.map:.4f}")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"Precision: {val_results.box.mp:.4f}")
    print(f"Recall: {val_results.box.mr:.4f}")

    cm = getattr(val_results, 'confusion_matrix', None)
    if cm is not None:
        cm_array = cm.matrix.astype(np.int32)
        TP = np.trace(cm_array)
        total = np.sum(cm_array)
        acc = TP / total if total > 0 else 0
        print(f"Classification Accuracy (Test): {acc:.4f}")

        plt.figure(figsize=(8, 8))
        plt.imshow(cm_array, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("YOLOv8 Confusion Matrix")
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig("/kaggle/working/yolov8_confusion_matrix.png")
        plt.show()
        print("[info] Saved confusion matrix plot and numpy array")
        np.save("/kaggle/working/confusion_matrix.npy", cm_array)
    else:
        print("[warning] Confusion matrix not generated.")

    print(f"[info] Running inference on a test image")
    # update with a valid test image from your set
    TEST_IMAGE = "/kaggle/input/hydroponics-tomato/test/images/0302_JPG.rf.49fd3f23a55d27d48a33dc63fc0e5626.jpg"
    infer_results = base_model(TEST_IMAGE)
    infer_results[0].show()
    infer_results[0].save(filename="/kaggle/working/test_inference_result.jpg")
    print("[info] Saved test image inference result")

if __name__ == "__main__":
    main()

