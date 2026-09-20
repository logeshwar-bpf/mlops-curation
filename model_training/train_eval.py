import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple
import argparse
import random
import timm
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from timeit import default_timer as timer
from torchmetrics import ConfusionMatrix, Accuracy, Precision, Recall
from mlxtend.plotting import plot_confusion_matrix
import boto3
from botocore.exceptions import NoCredentialsError

# Import your custom modules
import data_setup
import engine

def save_plot(fig, save_path, filename):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save the figure
    filepath = os.path.join(save_path, filename)
    fig.savefig(filepath)
    plt.close(fig)

def delete_directory(directory):
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(directory)
        print(f"[INFO] Deleted cache directory: {directory}")
    else:
        print(f"[INFO] Cache directory does not exist: {directory}")

def upload_directory_to_s3(local_directory, bucket, s3_directory):
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "YOUR_AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "YOUR_AWS_SECRET_ACCESS_KEY")
    
    # Initialize the S3 client
    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
    print(f"[INFO] Starting upload of model output to S3 bucket: {bucket}")
    for root, _, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(s3_directory, relative_path).replace("\\", "/")

            try:
                print(f"[INFO] Uploading {local_path} to s3://{bucket}/{s3_path}")
                s3_client.upload_file(local_path, bucket, s3_path)
                print(f"[INFO] Uploaded {local_path} to s3://{bucket}/{s3_path}")

                # Delete the local file after uploading
                os.remove(local_path)
                print(f"[INFO] Deleted local file: {local_path}")
            except FileNotFoundError:
                print(f"[ERROR] The file was not found: {local_path}")
            except NoCredentialsError:
                print("[ERROR] Credentials not available.")
    print("[INFO] All files have been uploaded and local copies deleted successfully.")

def find_best_epoch(results_file):
    best_epoch = None
    best_test_loss = float('inf')
    with open(results_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('Epoch'):
                parts = line.split('|')
                epoch = int(parts[0].split(':')[1].strip())
                test_loss = float(parts[3].split(':')[1].strip())
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_epoch = epoch
    return best_epoch

def evaluate_and_plot_confusion_matrix(model, dataloader, class_names, save_path, filename, device='cuda'):
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    confmat_tensor = confmat(torch.tensor(y_pred), torch.tensor(y_true))

    accuracy = Accuracy(task='multiclass', num_classes=len(class_names))
    accuracy_score = accuracy(torch.tensor(y_pred), torch.tensor(y_true))

    precision = Precision(task='multiclass', average='macro', num_classes=len(class_names))
    precision_score = precision(torch.tensor(y_pred), torch.tensor(y_true))

    recall = Recall(task='multiclass', average='macro', num_classes=len(class_names))
    recall_score = recall(torch.tensor(y_pred), torch.tensor(y_true))

    fig, ax = plot_confusion_matrix(
        confmat_tensor.cpu().numpy(),
        class_names=class_names,
        figsize=(10, 7)
    )

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy_score:.2f} | Precision: {precision_score:.2f} | Recall: {recall_score:.2f}')

    save_plot(fig, save_path, filename)

def export_best_epoch_model(args, model, model_name, val_dataloader, class_names):
    # Parse the results file to find the best epoch
    results_file = os.path.join(args.save_dir, model_name, 'training_results.txt')
    best_epoch = find_best_epoch(results_file)
    
    if best_epoch is None:
        print(f"[ERROR] No valid epoch found in {results_file}.")
        return
    
    print(f"[INFO] Best epoch: {best_epoch}")

    # Load the best model checkpoint
    best_model_path = os.path.join(args.save_dir, model_name, f'model_epoch_{best_epoch}.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print(f"[ERROR] Model checkpoint not found: {best_model_path}")
        return

    model = model.eval().cuda()
    model_export = torch.nn.Sequential(model, torch.nn.Softmax(1))
    x = torch.ones((1, 3, args.image_size[0], args.image_size[1])).cuda()
    export_path = os.path.join(args.save_dir, model_name, f'{model_name}_best_model.onnx')
    torch.onnx.export(model_export, x, export_path, opset_version=14,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    print(f"[INFO] Best model from epoch {best_epoch} has been successfully exported to ONNX format at {export_path}.")

    # Evaluate the best epoch model and plot confusion matrix
    evaluate_and_plot_confusion_matrix(model, val_dataloader, class_names, os.path.join(args.save_dir, model_name), 'confusion_matrix_best.png')

def main(args):
    # Define the transformation pipeline for training data
    train_transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=2),  # Bicubic interpolation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create training and testing DataLoaders
    train_dataloader, val_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.val_dir,
        transform=train_transform,
        batch_size=args.batch_size
    )
    print(class_names)

    # Get the number of classes
    num_classes = len(class_names)  # Dynamically determine number of classes

    for model_name in args.models:
        print(f"Training and evaluating model: {model_name}")

        # Load the pre-trained model
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        # Get the number of input features for the classifier
        num_in_features = model.get_classifier().in_features
        
        # Define the new classifier
        model.fc = nn.Sequential(
            nn.BatchNorm1d(num_in_features),
            nn.Linear(in_features=num_in_features, out_features=512, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(in_features=512, out_features=256, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(in_features=256, out_features=num_classes, bias=False)
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Define loss and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Move the model to the target device
        model.to(device)

        # Start the timer
        start_time = timer()
        # Setup training and save the results
        results = engine.train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=val_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=args.epochs,
            device=device,
            save_dir=os.path.join(args.save_dir, model_name)
        )





        # End the timer and print out how long it took
        end_time = timer()
        print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

        # Evaluate the final trained model and plot confusion matrix
        evaluate_and_plot_confusion_matrix(model, val_dataloader, class_names, os.path.join(args.save_dir, model_name), 'confusion_matrix_final.png')

        # Save the final trained model
        torch.save(model.state_dict(), os.path.join(args.save_dir, model_name, f'{model_name}_model.pt'))

        # Export the final trained model to ONNX format
        model = model.eval().cuda()
        model_export = torch.nn.Sequential(model, torch.nn.Softmax(1))
        x = torch.ones((1, 3, args.image_size[0], args.image_size[1])).cuda()
        torch.onnx.export(model_export, x, os.path.join(args.save_dir, model_name, f'{model_name}_model.onnx'), opset_version=14,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})

        print(f"[INFO] Model {model_name} has been successfully saved and exported to ONNX format.")

        # Export the best epoch model to ONNX format and evaluate it
        export_best_epoch_model(args, model, model_name, val_dataloader, class_names)

        # Upload model output to S3 and delete local files
        upload_directory_to_s3(os.path.join(args.save_dir, model_name), 'max-annotations', f'kids_model_/3class_sample_2/{model_name}')

        # Delete the Hugging Face cache directory
        delete_directory(f"/home/ubuntu/.cache/huggingface/hub/models--timm--{model_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train and evaluate the model.")
    parser.add_argument("--train_dir", type=str, default="/path/to/train", help="Path to the training data directory.")
    parser.add_argument("--val_dir", type=str, default="/path/to/val", help="Path to the validation data directory.")
    parser.add_argument("--test_dir", type=str, default="/path/to/test", help="Path to the test data directory.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training.")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate for model regularization.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--image_size", nargs=2, type=int, default=[224, 224], help="Image size for training and evaluation.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of CPU workers for data loading.")
    parser.add_argument("--save_dir", type=str, default="/path/to/save", help="Directory to save trained model and plots.")
    parser.add_argument("--models", nargs="+", type=str, default=[], help="List of models to train and evaluate.")

    args = parser.parse_args()

    main(args)
