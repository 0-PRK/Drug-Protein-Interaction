from Bio import ExPASy
from Bio import SwissProt
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import matplotlib.pyplot as plt
import py3Dmol
import requests
import xml.etree.ElementTree as ET
import time
import streamlit as st

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

        amino_features = aa_composition
        other_features = np.concatenate([
            [properties['molecular_weight']],
            [properties['aromaticity']],
            [properties['instability_index']],
            [properties['isoelectric_point']],
            [properties['hydrophobicity']]
        ])
        
        # Combine features
        features = np.concatenate([
            aa_composition,
            [properties['molecular_weight']],
            [properties['aromaticity']],
            [properties['instability_index']],
            [properties['isoelectric_point']],
            [properties['hydrophobicity']]
        ])
        
        return features, amino_features, other_features
    
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
    # plt.show()
    return plt

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
            st.error(f"No PDB IDs found for UniProt ID {uniprot_id}.")
            print(f"No PDB IDs found for UniProt ID {uniprot_id}.")
            return

        print("3D model for the protein: ",uniprot_id)
        view = py3Dmol.view(width=800, height=600)
        view.addModel(pdb, "pdb")
        view.setStyle({"cartoon": {"color": "spectrum"}})
        view.zoomTo()
        # view.show()
        return view
        
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
        st.error("Error submitting sequence to BLAST.")
        print("Error submitting sequence to BLAST.")
        return None

    # Parse the request ID (RID)
    rid_start = response.text.find("RID = ") + len("RID = ")
    rid_end = response.text.find("\n", rid_start)
    rid = response.text[rid_start:rid_end].strip()
    st.write(f"BLAST request submitted. RID: {rid}")
    print(f"BLAST request submitted. RID: {rid}")

    status_text = st.empty()
    ti = 0

    # Step 2: Wait for the results to be ready
    while True:
        status_response = requests.get(blast_url, params={"CMD": "Get", "RID": rid, "FORMAT_TYPE": "XML"})
        if "Status=WAITING" not in status_response.text:
            break
        print("Waiting for BLAST results...")
        # status_text.text(f"Waiting for BLAST results.... Time elapsed:{ti}s")
        for i in range(5):
            status_text.text(f"Waiting for BLAST results.... Time elapsed:{ti}s")
            time.sleep(1)
            ti = ti+1

    # Step 3: Retrieve the results
    result_response = requests.get(blast_url, params={"CMD": "Get", "RID": rid, "FORMAT_TYPE": "XML"})
    if result_response.status_code != 200:
        print("Error retrieving BLAST results.")
        st.error("Error retrieving BLAST results.")
        return None

    # Step 4: Parse the XML results to extract UniProt ID
    try:
        root = ET.fromstring(result_response.text)
        hits = root.findall(".//Hit")
        for hit in hits:
            hit_def = hit.find("Hit_def").text
            if "sp|" in hit_def:  # Check for SwissProt entries
                uniprot_id = hit_def.split("|")[1]  # Extract UniProt ID
                # Strip version suffix (e.g., ".1") if present
                clean_uniprot_id = uniprot_id.split('.')[0]
                st.write(f"The uniprot_id found was: {clean_uniprot_id}")
                return clean_uniprot_id
    except Exception as e:
        st.error(f"Error parsing BLAST results: {e}")
        print(f"Error parsing BLAST results: {e}")

    print("No matching UniProt ID found.")
    st.warning("No matching UniProt ID found.")
    return None

    
def fetch_protein_info_uniprot(uniprot_id:str):

    # Strip version suffix (e.g., ".1") if present
    clean_uniprot_id = uniprot_id.split('.')[0]

    url = f"https://www.uniprot.org/uniprot/{clean_uniprot_id}.xml"
    response = requests.get(url)

    handle = ExPASy.get_sprot_raw(uniprot_id)
    record = SwissProt.read(handle)
    
    if response.status_code != 200:
        st.error(f"Error fetching UniProt entry for {uniprot_id}")
        raise Exception(f"Error fetching UniProt entry for {uniprot_id}")
    
    root = ET.fromstring(response.content)
    
    protein_name = root.find(".//{http://uniprot.org/uniprot}recommendedName/{http://uniprot.org/uniprot}fullName").text
    organism = root.find(".//{http://uniprot.org/uniprot}organism/{http://uniprot.org/uniprot}name[@type='scientific']").text
    gene_name = root.find(".//{http://uniprot.org/uniprot}gene/{http://uniprot.org/uniprot}name[@type='primary']").text
    subcellular_location = root.find(".//{http://uniprot.org/uniprot}comment[@type='subcellular location']/{http://uniprot.org/uniprot}subcellularLocation/{http://uniprot.org/uniprot}location")
    subcellular_location_text = subcellular_location.text if subcellular_location is not None else "Unknown"
    function = root.find(".//{http://uniprot.org/uniprot}comment[@type='function']/{http://uniprot.org/uniprot}text")
    function_text = function.text if function is not None else "Function not available"
    sequence = get_protein_from_uniprot(uniprot_id=uniprot_id)

    extractor = ProteinFeatureExtractor()

    basic_props = extractor.calculate_protein_properties(sequence=sequence)

    return {
        "Protein Name": protein_name,
        "Organism": organism,
        "Molecular Weight":basic_props.get('molecular_weight','N/A'),
        "Aromaticity":basic_props.get('aromaticity','N/A'),
        "Instability Index":basic_props.get('instability_index','N/A'),
        "IsoElectric Point":basic_props.get('isoelectric_point','N/A'),
        "Hydrophobicity":basic_props.get('hydrophobicity','N/A'),
        "Keywords":record.keywords,
        "Sequence Length":record.sequence_length,
        "Sequence":sequence,
        "Protein Existence": record.protein_existence,
        "Gene Name": gene_name,
        "Subcellular Location": subcellular_location_text,
        "Function": function_text,
        "Description":record.description
    }

def fetch_protein_info_seq(sequence:str):
    uniprot_id = fetch_uniprot_id(sequence=sequence)
    if uniprot_id:

        info = fetch_protein_info_uniprot(uniprot_id)
        return info, uniprot_id
    else:
        raise Exception(f"Error finding the uniprot_id for {sequence}")
    
def parse_protein_description(description: str):
    # Split the description into parts based on the delimiters
    parts = description.split("; ")
    parsed_info = {}

    # Process each part
    for part in parts:
        # Remove evidence tags like {ECO:...}
        clean_part = part.split(" {")[0]
        
        # Check for specific patterns
        if clean_part.startswith("RecName: Full="):
            parsed_info["Recommended Name"] = clean_part.replace("RecName: Full=", "").strip()
        elif clean_part.startswith("Short="):
            parsed_info["Short Name"] = clean_part.replace("Short=", "").strip()
        elif clean_part.startswith("EC="):
            parsed_info["EC Number"] = clean_part.replace("EC=", "").strip()
        elif clean_part.startswith("AltName: Full="):
            parsed_info["Alternative Name"] = clean_part.replace("AltName: Full=", "").strip()
        else:
            # Other unclassified data
            parsed_info.setdefault("Other Details", []).append(clean_part)

    return parsed_info

import plotly.express as px

def box_plot(sequence: str):
    
    extractor = ProteinFeatureExtractor()
    features, amino_features, other_features = extractor.extract_features(sequence)
    feature_labels_1 = extractor.amino_acids
    feature_labels_2 = [
        "Molecular Weight",
        "Aromaticity",
        "Instability Index",
        "Isoelectric Point",
        "Hydrophobicity"
    ]

    if 'Molecular Weight' in feature_labels_2:
        molwt_index = feature_labels_2.index('Molecular Weight')
        other_features = [f for i, f in enumerate(other_features) if i != molwt_index]
        feature_labels_2 = [f for i, f in enumerate(feature_labels_2) if i != molwt_index]
    
    fig1 = px.box(
        x=feature_labels_1, 
        y=amino_features, 
        points="all",  # Show all data points
        title="Protein Feature Distribution (Amino Acids)"
    )
    fig2 = px.box(
        x=feature_labels_2, 
        y=other_features, 
        points="all",  # Show all data points
        title="Protein Feature Distribution (Protein Features)"
    )
    return fig1, fig2

import plotly.graph_objects as go

def radar_chart(sequence: str):
    extractor = ProteinFeatureExtractor()
    features, amino_features, other_features = extractor.extract_features(sequence)
    feature_labels_1 = extractor.amino_acids
    feature_labels_2 = [
        "Molecular Weight",
        "Aromaticity",
        "Instability Index",
        "Isoelectric Point",
        "Hydrophobicity"
    ]

    if 'Molecular Weight' in feature_labels_2:
        molwt_index = feature_labels_2.index('Molecular Weight')
        other_features = [f for i, f in enumerate(other_features) if i != molwt_index]
        feature_labels_2 = [f for i, f in enumerate(feature_labels_2) if i != molwt_index]
    
    fig1 = go.Figure(data=[go.Scatterpolar(
        r=amino_features,
        theta=feature_labels_1,
        fill='toself',
        marker=dict(color='skyblue')
    )])
    fig1.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(amino_features)]  # Set range dynamically
            )
        ),
        title="Protein Feature Visualization (Radar Chart) - Amino Acids"
    )
    fig2 = go.Figure(data=[go.Scatterpolar(
        r=other_features,
        theta=feature_labels_2,
        fill='toself',
        marker=dict(color='skyblue')
    )])
    fig2.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(other_features)]  # Set range dynamically
            )
        ),
        title="Protein Feature Visualization (Radar Chart) - Protein Features"
    )
    return fig1, fig2

