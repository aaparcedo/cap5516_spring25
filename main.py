import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, models
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from tqdm import tqdm 
from sklearn.metrics import classification_report


# seed all random number generators for reproducibility
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# make sure results directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# used to denormalize images (model output is normalized)
def denormalize_image(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

# plots training and validation loss curves
def plot_loss(train_losses, val_losses, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

# plot failure cases 
def plot_failure(samples, class_names, mean, std, output_dir):
    plt.figure(figsize=(12, 4))  
    for i, sample in enumerate(samples[:4]): 
        img, true_label, pred_label = sample
        img_denorm = denormalize_image(img, mean, std)
        img_pil = to_pil_image(img_denorm)
        plt.subplot(1, 4, i + 1)
        plt.imshow(img_pil)
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}")
        plt.axis("off")
    plt.tight_layout()
    plt.suptitle("Failure Samples", fontsize=16, y=0.1)
    plt.savefig(os.path.join(output_dir, "failure.png"))
    plt.close()

# use Grad-CAM to visualize failure or success cases
def visualize_gradcam(model, cam_extractor, samples, class_names, mean, std, output_dir, is_success=True):
    plt.figure(figsize=(12, 4))
    success_or_failure = "Success" if is_success else "Failure"
    title = f"Grad-Cam {success_or_failure} Cases from ResNet-18 4th/Last Residual Block."
    filtered_samples = [s for s in samples if (s[1] == s[2]) == is_success][:4]  # First 4 success/failure samples

    for i, sample in enumerate(filtered_samples):
        img, true_label, pred_label = sample
        img = img.unsqueeze(0).to(device)
        model.eval()
        with torch.set_grad_enabled(True):
            output = model(img)
            _, predicted = output.max(1)
        activation_map = cam_extractor(predicted.item(), output)
        img_denorm = denormalize_image(img.squeeze().cpu(), mean, std)
        pil_img = to_pil_image(img_denorm)
        heatmap = overlay_mask(pil_img, to_pil_image(activation_map[0], mode="F"), alpha=0.5)
        plt.subplot(1, 4, i + 1)
        plt.imshow(heatmap)
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}")
        plt.axis("off")
    plt.tight_layout()
    filename = "gradcam_success.png" if is_success else "gradcam_failure.png"
    plt.suptitle(title, fontsize=16, y=0.1)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# train, save best checkpoint according to validation loss 
def train_model(model, criterion, optimizer, scheduler, trainloader, valloader, num_epochs, patience, output_dir):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(output_dir, "best_model.pth")

    # training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(trainloader))

        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(valloader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(valloader)
        val_losses.append(avg_val_loss)
        scheduler.step()
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    plot_loss(train_losses, val_losses, output_dir)
    return best_model_path

# test model with best checkpoint according to validation loss
def test_model(model, testloader, class_names):
    model.eval()
    y_true, y_pred = [], []
    failure_samples = []
    success_samples = []
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Testing Model"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
            for i in range(labels.size(0)):
                if predicted[i] != labels[i]:
                    failure_samples.append((inputs[i].cpu(), labels[i].item(), predicted[i].item()))
                else:
                    success_samples.append((inputs[i].cpu(), labels[i].item(), predicted[i].item()))
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return failure_samples, success_samples

def main():
    
    CXP_DATASET_PATH = "~/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray"
    BCXP_DATASET_PATH = "~/cap5516_spring25/balanced_pneumonia"

    seed_all(42)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    num_epochs = 20
    patience = 5
    learning_rate = 0.0001
    step_size = 10
    gamma = 0.1
    num_classes = 2

    cases = [
        {"dataset_path": CXP_DATASET_PATH, "pretrained": False, "augmentation": False, "case_name": "Case1"},
        {"dataset_path": BCXP_DATASET_PATH, "pretrained": False, "augmentation": True, "case_name": "Case2"},
        {"dataset_path": CXP_DATASET_PATH, "pretrained": True, "augmentation": False, "case_name": "Case3"},
        {"dataset_path": BCXP_DATASET_PATH, "pretrained": True, "augmentation": True, "case_name": "Case4"},
    ]

    for case in cases:
        output_dir = os.path.join("results", case["case_name"])
        
        ensure_dir(output_dir)

        dataset_path = case["dataset_path"]
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)])
        trainset = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform)
        valset = datasets.ImageFolder(os.path.join(dataset_path, "val"), transform)
        testset = datasets.ImageFolder(os.path.join(dataset_path, "test"), transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)
        valloader = torch.utils.data.DataLoader(valset, batch_size=256, shuffle=False, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)

        model = models.resnet18(pretrained=case["pretrained"])
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

        best_model_path = train_model(model, nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=learning_rate), 
                                      optim.lr_scheduler.StepLR(optim.Adam(model.parameters(), lr=learning_rate), step_size=step_size, gamma=gamma),
                                      trainloader, valloader, num_epochs, patience, output_dir)

        model.load_state_dict(torch.load(best_model_path))
        failure_samples, success_samples = test_model(model, testloader, trainset.classes)
        plot_failure(failure_samples[:4], trainset.classes, mean, std, output_dir)
        cam_extractor = GradCAM(model, target_layer="layer4")
        visualize_gradcam(model, cam_extractor, success_samples, trainset.classes, mean, std, output_dir, is_success=True)
        visualize_gradcam(model, cam_extractor, failure_samples, trainset.classes, mean, std, output_dir, is_success=False)

if __name__ == "__main__":
    main()
