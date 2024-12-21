from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import py3Dmol
import pubchempy as pcp

def get_smiles_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        molecular_weight = Descriptors.MolWt(mol)
        logP = Descriptors.MolLogP(mol)
        return {'molecular_weight': molecular_weight, 'logP': logP}
    return None

def get_drug_details(smiles):
    # Fetch compound details from PubChem
    try:
        compound = pcp.get_compounds(smiles, 'smiles')[0]
    except IndexError:
        print("No compound found for the given SMILES.")
        return None

    # Extract compound details
    details = {
        "Name": compound.iupac_name,
        "Molecular Formula": compound.molecular_formula,
        "Molecular Weight": compound.molecular_weight,
        "Canonical SMILES": compound.canonical_smiles,
        "InChI": compound.inchi,
        "InChI Key": compound.inchikey,
        "Charge": compound.charge,
        "Exact Mass": compound.exact_mass,
        "XLogP": compound.xlogp,
    }

    return details, compound.canonical_smiles

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
        
def visualize_molecule_3d(smiles):
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Add hydrogens and generate 3D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Convert molecule to MOL block
    mol_block = Chem.MolToMolBlock(mol)
    
    # Visualize using py3Dmol
    viewer = py3Dmol.view(width=400, height=400)
    viewer.addModel(mol_block, "mol")
    viewer.setStyle({"stick": {}})
    viewer.zoomTo()
    return viewer

def extract_features(smilesList:list):
    extractor = DrugFeatureExtractor()
    data = []
    for smiles in smilesList:
        try:
            features = extractor.extract_features(smiles)
            data.append(features[:len(extractor.feature_names)])  # Exclude fingerprint
        except ValueError as e:
            print(f"Error with SMILES {smiles}: {e}")
    feature_df = pd.DataFrame(data, columns=extractor.feature_names)
    return feature_df

def visualize_drugs(smilesList:list):
    for smiles in smilesList:
        try:
            print(f"3D visualization for {smiles}:")
            viewer = visualize_molecule_3d(smiles)
            viewer.show()
        except ValueError as e:
            print(f"Error with SMILES {smiles}: {e}")

def feature_dist_and_feature_corr(feature_df:pd.DataFrame):
    feature_df.hist(figsize=(10, 8), bins=15, color='skyblue', edgecolor='black')
    plt.suptitle("Feature Distributions")
    plt.show()

    # Visualization: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(feature_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

def morgan_fingerprint_heatmap(smilesList:list):
    extractor = DrugFeatureExtractor()
    for smiles in smilesList:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            morgan_fp = extractor.get_morgan_fingerprint(mol)
            plt.figure(figsize=(12, 2))
            sns.heatmap([morgan_fp], cmap='viridis', cbar=False, xticklabels=False, yticklabels=[smiles])
            plt.title(f"Morgan Fingerprint for {smiles}")
            plt.show()





