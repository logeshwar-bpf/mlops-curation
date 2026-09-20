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

import os
import shutil

def delete_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Delete the directory
        shutil.rmtree(directory_path)

        print(f"[INFO] Deleted cache directory: {directory_path}")
    else:
        print(f"[INFO] Cache directory does not exist: {directory_path}")

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
                #os.remove(local_path)
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

def export_best_epoch_model(args, model, model_name):
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



def main(args):
    # Define the transformation pipeline for training data
    #delete_directory(os.path.expanduser(f"~/.cache/huggingface/hub/models--timm--{model_name}"))
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.transforms as transforms
    import torch
    import numpy as np
    from PIL import Image, ImageOps
    import torchvision.transforms as transforms
    '''class PadToSquareAndResize:
        def __init__(self, size, fill=0):
            self.size = size
            self.fill = fill
    
        def __call__(self, img):
            # Add padding to make the image square
            img = ImageOps.pad(img, (self.size, self.size), color=(self.fill, self.fill, self.fill))
            return img'''
    train_transform = transforms.Compose([
        #PadToSquareAndResize(224),
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
        
        # Define the new classifier with correct dimensions
        custom_classifier = nn.Sequential(
            nn.BatchNorm1d(num_in_features),
            nn.Linear(in_features=num_in_features, out_features=512, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(args.dropout),
            nn.Linear(in_features=512, out_features=256, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # Fixed: was 512, should be 256
            nn.Dropout(args.dropout),
            nn.Linear(in_features=256, out_features=num_classes, bias=True)  # Added bias=True for final layer
        )
        
        # Replace classifier based on model architecture
        model_name_lower = model_name.lower()
        
        if 'convnext' in model_name_lower:
            # ConvNeXt has head.fc as the final classifier, keep the pooling/norm layers
            if hasattr(model, 'head') and hasattr(model.head, 'fc'):
                model.head.fc = custom_classifier
            else:
                # Fallback: reset head with proper pooling
                model.reset_classifier(num_classes=0)
                model.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1),
                    nn.BatchNorm1d(num_in_features),
                    nn.Linear(in_features=num_in_features, out_features=512, bias=False),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(args.dropout),
                    nn.Linear(in_features=512, out_features=256, bias=False),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(args.dropout),
                    nn.Linear(in_features=256, out_features=num_classes, bias=True)
                )
        elif 'vit' in model_name_lower or 'swin' in model_name_lower or 'deit' in model_name_lower:
            # Vision Transformers typically use model.head
            model.head = custom_classifier
        elif hasattr(model, 'fc'):
            # ResNet, etc.
            model.fc = custom_classifier
        elif hasattr(model, 'classifier'):
            # EfficientNet, MobileNet, etc.
            model.classifier = custom_classifier
        elif hasattr(model, 'head'):
            # Generic head replacement
            model.head = custom_classifier
        else:
            # Last resort: use timm's reset_classifier and set fc
            print(f"[WARNING] Unknown model architecture {model_name}, attempting generic classifier replacement")
            model.reset_classifier(num_classes=num_classes)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #class_weights = torch.tensor([5.529536361496823, 2.6557024980219284, 2.8420224990927787, 13.683750728013978, 56.6144578313253]).to(device)
        #(weight=class_weights)

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

        # Evaluate the model
        transform = transforms.Compose([
            #PadToSquareAndResize(224),
            transforms.Resize(args.image_size, interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_dataset = datasets.ImageFolder(root=args.test_dir, transform=transform)
        testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        y_true = []
        y_pred = []

        model.eval()
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

        classes = class_names

        confmat = ConfusionMatrix(num_classes=num_classes, task='multiclass')
        confmat_tensor = confmat(torch.tensor(y_pred), torch.tensor(y_true))

        accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        accuracy_score = accuracy(torch.tensor(y_pred), torch.tensor(y_true))

        precision = Precision(task='multiclass', average='macro', num_classes=num_classes)
        precision_score = precision(torch.tensor(y_pred), torch.tensor(y_true))

        recall = Recall(task='multiclass', average='macro', num_classes=num_classes)
        recall_score = recall(torch.tensor(y_pred), torch.tensor(y_true))

        fig, ax = plot_confusion_matrix(
            confmat_tensor.cpu().numpy(),
            class_names=classes,
            figsize=(10, 7)
        )

        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix\nAccuracy: {accuracy_score:.2f} | Precision: {precision_score:.2f} | Recall: {recall_score:.2f}')

        save_plot(fig, os.path.join(args.save_dir, model_name), 'confusion_matrix.png')

        print(f'Accuracy: {accuracy_score:.2f}')
        print(f'Precision: {precision_score:.2f}')
        print(f'Recall: {recall_score:.2f}')
        def pred_and_plot_image(model: torch.nn.Module,
                                image_path: str,
                                class_names: List[str],
                                image_size: Tuple[int, int] = args.image_size,
                                transform: transforms.Compose = transform,
                                device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> Tuple[Image.Image, str, float]:
            # Open image
            img = Image.open(image_path).convert('RGB')

            # Create transformation for image (if one doesn't exist)
            if transform is None:
                transform = transforms.Compose([
                    #PadToSquareAndResize(224),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

            # Ensure the model is on the target device
            model.to(device)

            model.eval()
            with torch.inference_mode():
                # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
                transformed_image = transform(img).unsqueeze(dim=0)

                # Make a prediction on image with an extra dimension and send it to the target device
                target_image_pred = model(transformed_image.to(device))

            # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
            target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

            # Convert prediction probabilities -> prediction labels
            target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

            # Extract prediction label and probability
            pred_label = class_names[target_image_pred_label.item()]
            pred_prob = target_image_pred_probs.max().item()

            return img, pred_label, pred_prob

        def plot_images_horizontally(image_data: List[Tuple[Image.Image, str, float]], title: str, save_path: str):
            fig, axs = plt.subplots(1, len(image_data), figsize=(15, 5))
            fig.suptitle(title, fontsize=16)
            if len(image_data) == 1:
                axs = [axs]
            for ax, (img, pred_label, pred_prob) in zip(axs, image_data):
                ax.imshow(img)
                ax.set_title(f"Pred: {pred_label}\nProb: {pred_prob:.3f}")
                ax.axis('off')
            plt.tight_layout()

            # Save the plot
            save_plot(fig, save_path, f"{title}.png")

        def save_plots_with_different_prediction(image_data: List[Tuple[Image.Image, str, float]], subfolder: str, save_path: str):
            incorrect_images = [(img, pred_label, pred_prob) for img, pred_label, pred_prob in image_data if pred_label != subfolder]
            incorrect_images = incorrect_images[:7]
            if incorrect_images:
                fig, axs = plt.subplots(1, len(incorrect_images), figsize=(15, 5))
                fig.suptitle(subfolder, fontsize=16)
                if len(incorrect_images) == 1:
                    axs = [axs]
                for ax, (img, pred_label, pred_prob) in zip(axs, incorrect_images):
                    ax.imshow(img)
                    ax.set_title(f"Pred: {pred_label}\nProb: {pred_prob:.3f}")
                    ax.axis('off')
                plt.tight_layout()
                # Save the plot under "wrong_subfolder"
                save_plot(fig, save_path, f"wrong_{subfolder}.png")

        for subfolder in os.listdir(args.test_dir):
            subfolder_path = os.path.join(args.test_dir, subfolder)
            if os.path.isdir(subfolder_path):
                image_files = [f for f in os.listdir(subfolder_path) if f.endswith((".png", ".jpg", ".bmp"))]

                # Predict all images but only plot up to 5
                image_data = []
                for image_file in image_files:
                    image_path = os.path.join(subfolder_path, image_file)
                    img, pred_label, pred_prob = pred_and_plot_image(model=model,
                                                                    image_path=image_path,
                                                                    class_names=class_names,
                                                                    transform=train_transform)
                    image_data.append((img, pred_label, pred_prob))

                # Plot only up to 5 images
                plot_images = image_data[:5]
                plot_images_horizontally(plot_images, title=subfolder, save_path=os.path.join(args.save_dir, model_name))
                
                # Save plots with different predictions for all images
                save_plots_with_different_prediction(image_data, subfolder, save_path=os.path.join(args.save_dir, model_name))

        torch.save(model.state_dict(), os.path.join(args.save_dir, model_name, f'{model_name}_model.pt'))

        model = model.eval().cuda()
        model_export = torch.nn.Sequential(model, torch.nn.Softmax(1))
        x = torch.ones((1, 3, args.image_size[0], args.image_size[1])).cuda()
        torch.onnx.export(model_export, x, os.path.join(args.save_dir, model_name, f'{model_name}_model.onnx'), opset_version=14,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})

        print(f"[INFO] Model {model_name} has been successfully saved and exported to ONNX format.")

        export_best_epoch_model(args, model, model_name)

        # Upload model output to S3 and delete local files
        # upload_directory_to_s3(os.path.join(args.save_dir, model_name), 'max-annotations', f'jwc_staff/jwc_set_F_5/{model_name}')

        # Delete the Hugging Face cache directory
        delete_directory(os.path.expanduser(f"~/.cache/huggingface/hub/models--timm--{model_name}"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train and evaluate the model.")
    parser.add_argument("--train_dir", type=str, default="/kaggle/working/age_aug/train", help="Path to the training data directory.")
    parser.add_argument("--val_dir", type=str, default="/kaggle/working/age_aug/valid", help="Path to the validation data directory.")
    parser.add_argument("--test_dir", type=str, default="/kaggle/working/age_aug/valid", help="Path to the test data directory.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for training.")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate for model regularization.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--image_size", nargs=2, type=int, default=[224, 224], help="Image size for training and evaluation.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of CPU workers for data loading.")
    parser.add_argument("--save_dir", type=str, default="/kaggle/working", help="Directory to save trained model and plots.")
    parser.add_argument("--models", nargs="+", type=str, default=[], help="List of models to train and evaluate.")

    args = parser.parse_args()

    main(args)


