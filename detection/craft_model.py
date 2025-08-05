# weights 
import torch
from models.craft_architecture import CRAFT  # Make sure this path is correct
def load_craft_model(weight_path, device='cpu'):
    print("Loading CRAFT weights from", weight_path)

    model = CRAFT()  # Instantiate model
    model.to(device)

    # Load weights
    state_dict = torch.load(weight_path, map_location=device, weights_only=False)

    # If the weights were trained using DataParallel, they will have a 'module.' prefix
    if list(state_dict.keys())[0].startswith("module."):
        print("Removing 'module.' prefix from keys...")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()

    print("CRAFT model loaded successfully.")
    return model
if __name__ == "__main__":
    model = load_craft_model("weights/craft_mlt_25k.pth", device='cuda' if torch.cuda.is_available() else 'cpu')
    print("Model is ready to use.")
