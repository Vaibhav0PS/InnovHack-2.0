from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("yolo11n.pt")

# Export the model
model.export(format="openvino")  # creates 'yolo11n_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO("yolo11n_openvino_model/")

# Run inference
results = ov_model("https://ultralytics.com/images/bus.jpg")

# Run inference with specified device, available devices: ["intel:gpu", "intel:npu", "intel:cpu"]
results = ov_model("https://ultralytics.com/images/bus.jpg", device="intel:gpu")