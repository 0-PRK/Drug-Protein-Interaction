import streamlit as st
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import matplotlib.pyplot as plt
import py3Dmol
import requests
import numpy as np
from io import StringIO
from inference import inference
import streamlit.components.v1 as components
import pandas as pd

# Import your custom classes and functions
from protein import (
    get_protein_from_uniprot,
    ProteinFeatureExtractor,
    feature_visualization,
    visualize_protein_3d,
    fetch_uniprot_id,
    fetch_protein_info_uniprot,
    fetch_protein_info_seq,
    parse_protein_description,
    box_plot,
    radar_chart
)

from drug import (
    DrugFeatureExtractor,
    get_drug_details,
    extract_features,
    feature_dist_and_feature_corr,
    morgan_fingerprint_heatmap,
    visualize_molecule_3d,
    interactive_radar,
    bar_plot,
    pie_chart,
    interactive_bar,
)

def main():
    st.title("Protein Analysis and Visualization")
    st.write("Analyze and visualize protein features and interactions.")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Go to", ["Home","Drug-Protein Interaction", "Protein Analysis", "Drug Analysis"])

    if menu == "Drug-Protein Interaction":
        st.header("Drug-Protein Interaction Prediction")
        smiles = st.text_input("Enter SMILES String:")
        sequence = st.text_area("Enter Protein Sequence:")
        if st.button("Predict Interaction"):
            if smiles and sequence:
                try:
                    # Replace with your inference function
                    predicted_class, confidence_scores = inference(smiles, sequence)
                    st.success("Prediction Results")
                    st.write(f"Predicted Class: {predicted_class[0]}")
                    st.write(f"Confidence Score: {confidence_scores[0]:.4f}")
                    if predicted_class[0] == 0:
                        st.write("The drug is predicted to not bind to the given protein.")
                        st.write(f"Percentage Probability of not binding: {((1-confidence_scores[0])*100):.4f}%")
                    else:
                        st.write("The drug is predicted to bind to the given protein.")
                        st.write(f"Percentage Probability of binding: {((confidence_scores[0])*100):.4f}%")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please provide both SMILES and Protein Sequence.")
    elif menu == "Home":
        st.title("AI-Powered Drug-Protein Interaction Predictor")

        st.header("Introduction")
        st.write("""
            Welcome to the **AI-Powered Drug-Protein Interaction Predictor**! This project uses machine learning
            models to predict potential interactions between a given drug and a protein. You can input protein sequences
            or Uniprot IDs, and the model will predict whether a given drug interacts with the provided protein.
        """)

        st.header("How It Works")
        st.write("""
            This tool leverages deep learning to analyze protein and drug data. The input can either be a protein sequence
            or an Uniprot ID. The tool then uses a trained model to predict the likelihood of a drug-protein interaction.
        """)


    elif menu == "Protein Analysis":
        
        menu1 = st.selectbox("Choose input type:", ["UniProt ID", "Protein Sequence"])
        chosen_input = st.text_input("Enter the above mentioned input:")
        st.header("Fetch Protein Info")
        if st.button("Fetch Info"):
            if chosen_input:
                try:
                    # Call appropriate function based on input type
                    if menu1 == "UniProt ID":
                        info = fetch_protein_info_uniprot(chosen_input)  # Your function
                        sequence = get_protein_from_uniprot(chosen_input)
                        uniprot_id = chosen_input
                    else:
                        info, uniprot_id = fetch_protein_info_seq(chosen_input)
                        sequence = chosen_input
                    if not info:
                        st.error("Error while fetching the information.")
                        return

                    # Display Protein Info
                    st.subheader("Protein Details")
                    st.write(f"**UniProt ID:** {uniprot_id}")
                    st.write(f"**Protein Name:** {info.get('Protein Name', 'N/A')}")
                    st.write(f"**Organism:** {info.get('Organism', 'N/A')}")
                    st.write(f"**Gene Name:** {info.get('Gene Name', 'N/A')}")
                    st.write(f"**Molecular Weight:** {info.get('Molecular Weight', 'N/A')}")
                    st.write(f"**Aromaticity:** {info.get('Aromaticity', 'N/A')}")
                    st.write(f"**Instability Index:** {info.get('Instability Index', 'N/A')}")
                    st.write(f"**IsoElectric Point:** {info.get('IsoElectric Point', 'N/A')}")
                    st.write(f"**Hydrophobicity:** {info.get('Hydrophobicity', 'N/A')}")
                    st.write(f"**Protein Existence:** {info.get('Protein Existence', 'N/A')}")
                    st.write(f"**Subcellular Location:** {info.get('Subcellular Location', 'N/A')}")
                    st.write(f"**Sequence Length:** {info.get('Sequence Length', 'N/A')} residues")
                    st.write(f"**Sequence:** {info.get('Sequence', 'N/A')}")
                    st.write(f"**Function:** {info.get('Function', 'N/A')}")
                    
                    # Display Keywords
                    st.write("**Keywords:**")
                    keywords = info.get('Keywords', [])
                    if keywords:
                        for keyword in keywords:
                            st.markdown(f"- {keyword}")
                    else:
                        st.write("No keywords available.")

                    if "Description" in info:
                        st.subheader("Protein Description")

                        description = info["Description"]
                        parsed_description = parse_protein_description(description)

                        for key, value in parsed_description.items():
                            if isinstance(value, list):
                                st.write(f"**{key}:**")
                                for detail in value:
                                    st.markdown(f"- {detail}")
                            else:
                                st.write(f"**{key}:** {value}")

                except Exception as e:
                    st.error(f"Error fetching protein information: {e}")
            else:
                st.warning("Please enter a valid input.")

        st.header("Protein Feature Analysis")
        if st.button("Analyze Features"):
            if chosen_input:
                if menu1 == "UniProt ID":
                    sequence = get_protein_from_uniprot(chosen_input)
                else:
                    sequence = chosen_input
                
                if sequence:
                    try:
                        st.subheader("Box Plot for the Protein Features")
                        fig1, fig2 = box_plot(sequence=sequence)
                        st.plotly_chart(fig1)
                        st.plotly_chart(fig2)

                        st.subheader("Radar Chart for the Protein Features")
                        fig1, fig2 = radar_chart(sequence=sequence)
                        st.plotly_chart(fig1)
                        st.plotly_chart(fig2)

                        st.success("Feature Analysis Complete")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.error("No Sequence Found.")
            else:
                st.warning("Please enter a protein sequence.")
        st.header("Protein 3D Visualization")
        if st.button("Visualize"):
            if menu1 == "Protein Sequence" and chosen_input:
                sequence = chosen_input
                st.info("Fetching UniProt ID via BLAST... Please wait.")
                uniprot_id = fetch_uniprot_id(sequence)
                if uniprot_id:
                    st.success(f"UniProt ID found: {uniprot_id}")
                else:
                    st.error("No UniProt ID found.")
            elif menu1 == "UniProt ID" and chosen_input:
                uniprot_id = chosen_input
                st.info("Fetching and rendering 3D structure... Please wait.")
            else:
                st.warning("Please enter either a UniProt ID or a protein sequence.")

            # If UniProt ID is found, proceed to visualize the 3D structure
            if uniprot_id:
                width = 800
                height = 600
                view = visualize_protein_3d(uniprot_id)
                try:
                    html = view._make_html()
                    components.html(html, width=width, height=height)
                except Exception as e:
                    st.error(f"Error generating 3D visualization: {str(e)}")

    elif menu == "Drug Analysis":

        smiles_input = st.text_input("Enter a SMILES string:")
        st.header("Fetch Drug Details")
        if st.button("Get Drug Details"):
            if smiles_input:
                details, canonical_smiles = get_drug_details(smiles_input)
                if details:
                    st.subheader("Drug Details")
                    for key, value in details.items():
                        st.write(f"**{key}:** {value}")

                    st.subheader("Molecular Features")
                    extractor = DrugFeatureExtractor()
                    features = extractor.extract_features(canonical_smiles)
                    feature_df = pd.DataFrame([features[:len(extractor.feature_names)]], columns=extractor.feature_names)
                    st.dataframe(feature_df)
                else:
                    st.error("Could not fetch drug details. Please check the SMILES string.")
        
        st.header("Drug Feature Analysis")
        if st.button("Analyze Drug"):
            if smiles_input:
                smiles_list = []
                smiles_list.append(smiles_input)
                feature_df = extract_features(smiles_list)
                features = feature_df.iloc[0].values

                extractor = DrugFeatureExtractor()

                st.subheader("Extracted Features")
                st.dataframe(feature_df)
                
                st.subheader("Interactive Radar Chart")
                fig1 = interactive_radar(features, extractor.feature_names)
                st.plotly_chart(fig1)

                st.subheader("Interactive Bar Plot")
                fig2 = interactive_bar(features, extractor.feature_names)
                st.plotly_chart(fig2)

            else:
                st.error("Enter the smile please.")
        
        st.header("3D Molecule Visualization")
        if st.button("Visualize drug molecule"):
            if smiles_input:
                view = visualize_molecule_3d(smiles_input)
                width = 800
                height = 600
                try:
                    html = view._make_html()
                    components.html(html, width=width, height=height)
                except Exception as e:
                    st.error(f"Error generating 3D visualization: {str(e)}")
            else:
                st.error("Enter the smile please.")            

if __name__ == "__main__":
    main()
