from Bio import ExPASy
from Bio import SwissProt
# from transformers import ProteinBertModel
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np


def get_protein_from_uniprot(uniprot_id: str):
    handle = ExPASy.get_sprot_raw(uniprot_id)
    record = SwissProt.read(handle)
    return record.sequence


class ProteinFeatureExtractor:
    def __init__(self):
        # Define amino acid properties
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        self.aa_to_idx = {aa: idx for idx, aa in enumerate(self.amino_acids)}
        
        # Kyte-Doolittle hydrophobicity scale
        self.kd_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
    def sequence_to_vector(self, sequence):
        """Convert protein sequence to numerical vector"""
        # Initialize zero vector
        vector = np.zeros(len(self.amino_acids))
        
        # Count amino acid frequencies
        for aa in sequence:
            if aa in self.aa_to_idx:
                vector[self.aa_to_idx[aa]] += 1
                
        # Normalize by sequence length
        vector = vector / len(sequence)
        return vector
    
    def calculate_protein_properties(self, sequence):
        """Calculate basic protein properties"""
        try:
            analysis = ProteinAnalysis(sequence)
            
            # Calculate hydrophobicity using Kyte-Doolittle scale
            hydrophobicity = analysis.protein_scale(window=7, param_dict=self.kd_scale)
            
            properties = {
                'molecular_weight': analysis.molecular_weight(),
                'aromaticity': analysis.aromaticity(),
                'instability_index': analysis.instability_index(),
                'isoelectric_point': analysis.isoelectric_point(),
                'hydrophobicity': float(np.mean(hydrophobicity))  # Convert to float for consistency
            }
        except Exception as e:
            print(f"Error calculating protein properties: {e}")
            properties = {key: 0.0 for key in ['molecular_weight', 'aromaticity', 
                                             'instability_index', 'isoelectric_point', 
                                             'hydrophobicity']}
        return properties
    
    def extract_features(self, sequence):
        """Main feature extraction method"""
        # Basic input validation
        if not sequence or not isinstance(sequence, str):
            raise ValueError("Invalid protein sequence")
            
        # Clean sequence
        sequence = sequence.upper().strip()
        
        # Get amino acid composition
        aa_composition = self.sequence_to_vector(sequence)
        
        # Get protein properties
        properties = self.calculate_protein_properties(sequence)
        
        # Combine features
        features = np.concatenate([
            aa_composition,
            [properties['molecular_weight']],
            [properties['aromaticity']],
            [properties['instability_index']],
            [properties['isoelectric_point']],
            [properties['hydrophobicity']]
        ])
        
        return features
    
    
