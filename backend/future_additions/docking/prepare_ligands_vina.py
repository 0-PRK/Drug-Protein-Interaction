import os
import subprocess
from pathlib import Path

def prepare_ligand_for_vina(input_pdb, output_pdbqt, mgltools_path=None):
    """
    Prepare a ligand PDB file for AutoDock Vina using MGLTools' prepare_ligand4.py
    
    Parameters:
    input_pdb (str): Path to input PDB file
    output_pdbqt (str): Path to output PDBQT file
    mgltools_path (str): Path to MGLTools installation directory (optional)
    
    Returns:
    bool: True if conversion was successful, False otherwise
    """
    # Try to find MGLTools path if not provided
    if mgltools_path is None:
        common_paths = [
            '/home/aavash/Downloads/mgltools_x86_64Linux2_1.5.7p1/mgltools_x86_64Linux2_1.5.7',
            '/opt/mgltools',
            'C:\\Program Files\\MGLTools',
            os.path.expanduser('~/mgltools'),
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                mgltools_path = path
                break
        
        if mgltools_path is None:
            raise ValueError("MGLTools path not found. Please provide it explicitly.")

    # Construct path to prepare_ligand4.py
    prepare_ligand_script = Path(mgltools_path) / 'MGLToolsPckgs' / 'AutoDockTools' / 'Utilities24' / 'prepare_ligand4.py'
    
    if not prepare_ligand_script.exists():
        raise FileNotFoundError(f"prepare_ligand4.py not found at {prepare_ligand_script}")

    # Construct the command
    cmd = [
        'python',
        str(prepare_ligand_script),
        '-l', input_pdb,
        '-o', output_pdbqt,
        '-U', 'nphs',  # Merge non-polar hydrogens
        '-A', 'hydrogens',  # Add hydrogens
        '-C'  # Cleanup (remove water, compute Gasteiger charges)
    ]

    try:
        # Run the preparation script
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        if not os.path.exists(output_pdbqt):
            print(f"Error: Output file {output_pdbqt} was not created")
            print("Command output:", result.stdout)
            print("Command errors:", result.stderr)
            return False
            
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error running prepare_ligand4.py:")
        print(f"Command output: {e.stdout}")
        print(f"Command errors: {e.stderr}")
        return False

# Example usage
if __name__ == "__main__":
    # Example paths - update these according to your setup
    input_pdb = "ligand.pdb"
    output_pdbqt = "ligand_prepared.pdbqt"
    mgltools_path = "/usr/local/mgltools"  # Update this path
    
    success = prepare_ligand_for_vina(input_pdb, output_pdbqt, mgltools_path)
    if success:
        print("Ligand preparation completed successfully!")
    else:
        print("Ligand preparation failed!")