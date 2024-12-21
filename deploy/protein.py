from Bio import ExPASy
from Bio import SwissProt
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import matplotlib.pyplot as plt
import py3Dmol
import requests
import xml.etree.ElementTree as ET
import time

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
    
def feature_visualization(sequence:str):
    extractor = ProteinFeatureExtractor()
    features = extractor.extract_features(sequence)
    feature_labels = extractor.amino_acids + [
        "Molecular Weight",
        "Aromaticity",
        "Instability Index",
        "Isoelectric Point",
        "Hydrophobicity"
    ]

    plt.figure(figsize=(12, 6))
    plt.bar(feature_labels, features, color="skyblue", edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    plt.title("Protein Feature Visualization")
    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()

def get_protein_pdb(uniprot_id):
    """
        Check if a UniProt ID exists in the AlphaFold database and fetch the PDB file if available.
    """
    print("Checking in AlphaFold database:....")
    # Strip version suffix (e.g., ".1") if present
    clean_uniprot_id = uniprot_id.split('.')[0]

    alphafold_url = f"https://alphafold.ebi.ac.uk/files/AF-{clean_uniprot_id}-F1-model_v4.pdb"
    response = requests.get(alphafold_url)
        
    if response.status_code == 200:
        return response.text
    else:
        """Fetch PDB IDs associated with a UniProt ID using the UniProt API."""
        print("Not Found in AlphaFold. Checking in Uniprot.....")
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.xml"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Error fetching UniProt entry for {uniprot_id}")
    
        # Parse the XML response
        root = ET.fromstring(response.content)
        pdb_ids = []
        # Iterate through cross-references to find PDB entries
        for cross_ref in root.findall(".//{http://uniprot.org/uniprot}dbReference"):
            if cross_ref.attrib.get('type') == 'PDB':
                pdb_ids.append(cross_ref.attrib.get('id'))
        
        if not pdb_ids:
            print(f"No PDB IDs found for UniProt ID {uniprot_id}.")
            raise Exception(f"No PDB IDs found for UniProt ID {uniprot_id}.")
    
        pdb_id = pdb_ids[0]
    
        """Fetch the PDB file content from the PDB database."""
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Error fetching PDB file for {pdb_id}")

def visualize_protein_3d(uniprot_id: str):
    """Visualize the protein structure from a UniProt ID."""
    try:
        # Fetch PDB ID(s) associated with the UniProt ID
        pdb = get_protein_pdb(uniprot_id)
        
        if not pdb:
            print(f"No PDB IDs found for UniProt ID {uniprot_id}.")
            return

        print("3D model for the protein: ",uniprot_id)
        view = py3Dmol.view(width=800, height=600)
        view.addModel(pdb, "pdb")
        view.setStyle({"cartoon": {"color": "spectrum"}})
        view.zoomTo()
        view.show()
        
    except Exception as e:
        print(f"Error visualizing protein: {e}")

def fetch_uniprot_id(sequence):
    """
    Use the NCBI BLAST API to find the closest UniProt ID for the given protein sequence.
    """
    # Step 1: Submit the sequence to BLAST
    blast_url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
    params = {
        "CMD": "Put",
        "PROGRAM": "blastp",
        "DATABASE": "swissprot",
        "QUERY": sequence,
        "FORMAT_TYPE": "XML"
    }
    response = requests.post(blast_url, data=params)

    if response.status_code != 200:
        print("Error submitting sequence to BLAST.")
        return None

    # Parse the request ID (RID)
    rid_start = response.text.find("RID = ") + len("RID = ")
    rid_end = response.text.find("\n", rid_start)
    rid = response.text[rid_start:rid_end].strip()
    print(f"BLAST request submitted. RID: {rid}")

    # Step 2: Wait for the results to be ready
    while True:
        status_response = requests.get(blast_url, params={"CMD": "Get", "RID": rid, "FORMAT_TYPE": "XML"})
        if "Status=WAITING" not in status_response.text:
            break
        print("Waiting for BLAST results...")
        time.sleep(5)

    # Step 3: Retrieve the results
    result_response = requests.get(blast_url, params={"CMD": "Get", "RID": rid, "FORMAT_TYPE": "XML"})
    if result_response.status_code != 200:
        print("Error retrieving BLAST results.")
        return None

    # Step 4: Parse the XML results to extract UniProt ID
    try:
        root = ET.fromstring(result_response.text)
        hits = root.findall(".//Hit")
        for hit in hits:
            hit_def = hit.find("Hit_def").text
            if "sp|" in hit_def:  # Check for SwissProt entries
                uniprot_id = hit_def.split("|")[1]  # Extract UniProt ID
                return uniprot_id
    except Exception as e:
        print(f"Error parsing BLAST results: {e}")

    print("No matching UniProt ID found.")
    return None

    
