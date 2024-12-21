import torch
import torch.nn.functional as F
from test.model_prep.model import AttentionDTI  # Import the same model class used for training
from test.model_prep.hyperparameter import hyperparameter  # Load the same hyperparameters used for training
from test.model_prep.dataset import CHARISOSMISET, CHARPROTSET, label_smiles, label_sequence

# Step 1: Load hyperparameters and initialize the model
hp = hyperparameter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AttentionDTI(hp).to(device)

# Step 2: Load the saved model
model_path = "./valid_best_checkpoint.pth"  # Update path to your saved model file
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()  # Set the model to evaluation mode

MAX_SMI_LEN = 100
MAX_SEQ_LEN = 1000

def inference(smile, sequence):
    compound_input = torch.tensor(label_smiles(smile, CHARISOSMISET, MAX_SMI_LEN), dtype=torch.long).to(device)
    protein_input = torch.tensor(label_sequence(sequence, CHARPROTSET, MAX_SEQ_LEN), dtype=torch.long).to(device)

    compound_input = compound_input.unsqueeze(0)
    protein_input = protein_input.unsqueeze(0)

    # Step 4: Perform inference
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(compound_input, protein_input)

    # Step 5: Interpret the results
    probabilities = F.softmax(output, dim=1).cpu().numpy()
    predicted_class = probabilities.argmax(axis=1)  # Get the predicted class
    confidence_scores = probabilities[:, 1]  # Confidence score for the positive class

    return predicted_class, confidence_scores
    

