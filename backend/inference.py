import torch
import torch.nn.functional as F
from model import AttentionDTI  # Import the same model class used for training
# from dataset import preprocess_data  # Ensure the same preprocessing is used
from hyperparameter import hyperparameter  # Load the same hyperparameters used for training
from dataset import CHARISOSMISET, CHARPROTSET, label_smiles, label_sequence

# Step 1: Load hyperparameters and initialize the model
hp = hyperparameter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AttentionDTI(hp).to(device)

# Step 2: Load the saved model
model_path = "./KIBA/0/valid_best_checkpoint.pth"  # Update path to your saved model file
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()  # Set the model to evaluation mode

MAX_SMI_LEN = 100
MAX_SEQ_LEN = 1000

# Step 3: Prepare the data for inference
# Replace with your actual data preprocessing
compound_smiles = "C1COCCN1C2=CC(=CC=C2)NC3=NC=CC(=N3)C4=C(N=C5N4C=CS5)C6=CC(=CC=C6)NC(=O)CC7=CC=CC=C7"  # Your compound representation (SMILES string)
protein_sequence = "MLRGGRRGQLGWHSWAAGPGSLLAWLILASAGAAPCPDACCPHGSSGLRCTRDGALDSLHHLPGAENLTELYIENQQHLQHLELRDLRGLGELRNLTIVKSGLRFVAPDAFHFTPRLSRLNLSFNALESLSWKTVQGLSLQELVLSGNPLHCSCALRWLQRWEEEGLGGVPEQKLQCHGQGPLAHMPNASCGVPTLKVQVPNASVDVGDDVLLRCQVEGRGLEQAGWILTELEQSATVMKSGGLPSLGLTLANVTSDLNRKNVTCWAENDVGRAEVSVQVNVSFPASVQLHTAVEMHHWCIPFSVDGQPAPSLRWLFNGSVLNETSFIFTEFLEPAANETVRHGCLRLNQPTHVNNGNYTLLAANPFGQASASIMAAFMDNPFEFNPEDPIPVSFSPVDTNSTSGDPVEKKDETPFGVSVAVGLAVFACLFLSTLLLVLNKCGRRNKFGINRPAVLAPEDGLAMSLHFMTLGGSSLSPTEGKGSGLQGHIIENPQYFSDACVHHIKRRDIVLKWELGEGAFGKVFLAECHNLLPEQDKMLVAVKALKEASESARQDFQREAELLTMLQHQHIVRFFGVCTEGRPLLMVFEYMRHGDLNRFLRSHGPDAKLLAGGEDVAPGPLGLGQLLAVASQVAAGMVYLAGLHFVHRDLATRNCLVGQGLVVKIGDFGMSRDIYSTDYYRVGGRTMLPIRWMPPESILYRKFTTESDVWSFGVVLWEIFTYGKQPWYQLSNTEAIDCITQGRELERPRACPPEVYAIMRGCWQREPQQRHSIKDVHARLQALAQAPPVYLDVLG"  # Protein sequence

# Ensure tensors are on the correct device
compound_input = torch.tensor(label_smiles(compound_smiles, CHARISOSMISET, MAX_SMI_LEN), dtype=torch.long).to(device)
protein_input = torch.tensor(label_sequence(protein_sequence, CHARPROTSET, MAX_SEQ_LEN), dtype=torch.long).to(device)

compound_input = compound_input.unsqueeze(0)
protein_input = protein_input.unsqueeze(0)

# Step 4: Perform inference
with torch.no_grad():  # Disable gradient computation for inference
    output = model(compound_input, protein_input)

# Step 5: Interpret the results
probabilities = F.softmax(output, dim=1).cpu().numpy()
predicted_class = probabilities.argmax(axis=1)  # Get the predicted class
confidence_scores = probabilities[:, 1]  # Confidence score for the positive class

# Print results
print("Predicted Class:", predicted_class)
print("Confidence Scores:", confidence_scores)


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
    

