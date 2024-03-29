import torch

def checkpoint_model(model, optimizer, epoch, path):
    """
    Saves the model and optimizer state dictionaries to a checkpoint file.
    
    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        epoch (int): The current epoch number.
        path (str): The path to save the checkpoint file.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch} to {path}")

def load_model(model, optimizer, path, device, lr):
    """
    Loads a PyTorch model and optimizer from a checkpoint file.
    
    Args:
        model (torch.nn.Module): The PyTorch model to be loaded.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        path (str): The path to the checkpoint file.
        device (str or torch.device): The device to load the model on (e.g., 'cpu' or 'cuda').
    
    Returns:
        tuple: A tuple containing the loaded model, optimizer, and the epoch number.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print(f"Model loaded from {path} at epoch {epoch}")
    return model, optimizer, epoch
