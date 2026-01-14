import torch
import torchvision.transforms as transforms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def predict_on_test_images(
    trained_model,
    test_images,
    class_names,
    image_size: int = 224
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = trained_model.to(device)

    #turns off dropout and uses stored running stats for batchnorm
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    results = {}

    with torch.no_grad():
        for name, image in test_images.items():
            image = transform(image).unsqueeze(0).to(device)
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = probs.max(dim=1)
            results[name] = {
                "class": class_names[pred_idx.item()],
                "confidence": float(conf.item())
}
    return results