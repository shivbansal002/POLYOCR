import torch

def threshold_output(output, threshold=0.5):
    prob = torch.sigmoid(output)
    return (prob > threshold).float()

def decode_prediction(pred_tensor):
    # Customize based on your model's output
    return pred_tensor.cpu().numpy()

# TEST block
if __name__ == "__main__":
    # Simulate a raw output from a model
    raw_output = torch.tensor([[0.2, 0.8], [1.2, -0.4]])

    # Apply thresholding
    thresholded = threshold_output(raw_output, threshold=0.5)
    print("Thresholded Output:")
    print(thresholded)

    # Decode to numpy
    decoded = decode_prediction(thresholded)
    print("Decoded Output:")
    print(decoded)


