from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np


def get_smiles_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        molecular_weight = Descriptors.MolWt(mol)
        logP = Descriptors.MolLogP(mol)
        return {'molecular_weight': molecular_weight, 'logP': logP}
    return None


import requests

def get_drug_details(drugbank_id: str):
    url = f"https://api.drugbank.com/v1/drugs/{drugbank_id}"
    headers = {"Authorization": "Bearer YOUR_API_KEY"}
    response = requests.get(url, headers=headers)
    return response.json()  # Return the JSON response with drug details


class DrugFeatureExtractor:
    def __init__(self):
        self.feature_names = [
            'MolWt', 'LogP', 'NumRotatableBonds', 'AromaticProportion',
            'NumHAcceptors', 'NumHDonors', 'NumAromaticRings'
        ]
    
    def calculate_aromatic_proportion(self, mol):
        """Calculate proportion of aromatic atoms"""
        num_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        return num_aromatic / mol.GetNumAtoms()
    
    def get_morgan_fingerprint(self, mol, radius=2, nBits=1024):
        """Generate Morgan fingerprint and convert to numpy array safely"""
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        # Convert to numpy array safely using the explicit bit vector
        return np.array([int(morgan_fp.GetBit(i)) for i in range(morgan_fp.GetNumBits())])
    
    def extract_features(self, smiles):
        """Extract features from drug SMILES string"""
        try:
            # Convert SMILES to RDKit molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES string")
            
            # Calculate molecular descriptors
            features = {
                'MolWt': Descriptors.ExactMolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'AromaticProportion': self.calculate_aromatic_proportion(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol)
            }
            
            # Get Morgan fingerprint using the safe method
            morgan_features = self.get_morgan_fingerprint(mol)
            
            # Combine all features
            all_features = np.concatenate([
                [features[name] for name in self.feature_names],
                morgan_features
            ])
            
            return all_features
            
        except Exception as e:
            raise ValueError(f"Error processing drug SMILES: {e}")

