from protein import ProteinFeatureExtractor
from drug import DrugFeatureExtractor

def test_extractors():
    # Initialize extractors
    protein_extractor = ProteinFeatureExtractor()
    drug_extractor = DrugFeatureExtractor()
    
    # Test protein feature extraction
    test_sequence = "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTDGLLYGSQTPNEECLFLERLEENHYNTYISKKHAEKNWFVGLKKNGSCKRGPRTHYGQKAILFLPLPV"
    protein_features = protein_extractor.extract_features(test_sequence)
    
    # Test drug feature extraction
    test_smiles = "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"
    drug_features = drug_extractor.extract_features(test_smiles)
    
    return protein_features, drug_features

if __name__ == "__main__":
    protein_features, drug_features = test_extractors()
    print("Protein feature vector shape:", protein_features.shape)
    print(protein_features)
    print("Drug feature vector shape:", drug_features.shape)
    print(drug_features)