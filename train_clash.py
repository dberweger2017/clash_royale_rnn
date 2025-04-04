from ultralytics import YOLO
import torch # Optional: To check GPU availability

def main():
    # Check if GPU is available and print info
    if torch.cuda.is_available():
        print(f"CUDA is available. Training on GPU: {torch.cuda.get_device_name(0)}")
        device = 0 # Use the first available GPU
    else:
        print("CUDA not available. Training on CPU.")
        device = 'cpu'

    # --- Configuration ---
    # Path to your data.yaml file
    data_yaml_path = 'Annotated/data.yaml' # IMPORTANT: Change this!

    # Choose the model variant (yolov8s for small)
    model_variant = 'yolov8s.pt'

    # Training parameters
    epochs = 100         # Number of training epochs (adjust as needed)
    img_size = 640       # Input image size
    batch_size = 16      # Adjust based on your GPU memory (e.g., 8, 4 if 16 is too high)
    run_name = 'clash_royale_yolov8s_run1' # Name for the results folder

    # --- Load Model ---
    # Load a pre-trained YOLOv8s model
    # Using pre-trained weights significantly speeds up training (transfer learning)
    model = YOLO(model_variant)

    # --- Start Training ---
    print(f"Starting training with model: {model_variant}")
    print(f"Dataset configuration: {data_yaml_path}")
    print(f"Training for {epochs} epochs with image size {img_size} and batch size {batch_size}")

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name=run_name,      # Directory name for results (inside 'runs/detect/')
        device=device,      # Specify CPU or GPU ('0' for first GPU)
        patience=25,        # Early stopping patience (stop if no improvement after 25 epochs)
        exist_ok=True,      # Allows overwriting previous runs with the same name
        pretrained=True,    # Ensure using pre-trained weights
        optimizer='Adam',   # Optimizer choice
        lr0=0.001,          # Initial learning rate
        # Add other parameters if needed (see Ultralytics docs)
    )

    print("Training finished.")
    print(f"Results saved to: runs/detect/{run_name}")
    print(f"Best model weights saved at: runs/detect/{run_name}/weights/best.pt")

if __name__ == '__main__':
    main()
