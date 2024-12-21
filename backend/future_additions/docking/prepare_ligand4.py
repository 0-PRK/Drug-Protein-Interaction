#!/usr/bin/env python3

import os 
from MolKit import Read
from AutoDockTools.MoleculePreparation import AD4LigandPreparation

if __name__ == '__main__':
    import sys
    import getopt

    def usage():
        """Print helpful, accurate usage statement to stdout."""
        print("Usage: prepare_ligand4.py -l filename")
        print("")  # Python 3 empty print
        print("    Description of command...")
        print("         -l     ligand_filename (.pdb or .mol2 or .pdbq format)")
        print("    Optional parameters:")
        print("        [-v]    verbose output")
        print("        [-o pdbqt_filename] (default output filename is ligand_filename_stem + .pdbqt)")
        print("        [-d]    dictionary to write types list and number of active torsions ")
        print("        [-A]    type(s) of repairs to make:\n\t\t bonds_hydrogens, bonds, hydrogens (default is to do no repairs)")
        print("        [-C]    do not add charges (default is to add gasteiger charges)")
        print("        [-p]    preserve input charges on an atom type, eg -p Zn")
        print("               (default is not to preserve charges on any specific atom type)")
        print("        [-U]    cleanup type:\n\t\t nphs_lps, nphs, lps, '' (default is 'nphs_lps') ")
        print("        [-B]    type(s) of bonds to allow to rotate ")
        print("        [-R]    index for root")
        print("        [-F]    check for and use largest non-bonded fragment (default is not to do this)")
        print("        [-M]    interactive (default is automatic output)")
        print("        [-I]    string of bonds to inactivate composed of zero-based atom indices")
        print("        [-Z]    inactivate all active torsions")
        print("        [-g]    attach all nonbonded fragments")
        print("        [-s]    attach all nonbonded singletons")
        print("        [-w]    assign each ligand atom a unique name")

    # process command arguments
    try:
        opt_list, args = getopt.getopt(sys.argv[1:], 'l:vo:d:A:Cp:U:B:R:MFI:Zgswh')
    except getopt.GetoptError as msg:  # Python 3 exception syntax
        print('prepare_ligand4.py: %s' % msg)
        usage()
        sys.exit(2)

    # initialize required parameters
    ligand_filename = None
    verbose = None
    add_bonds = False
    repairs = ""
    charges_to_add = 'gasteiger'
    preserve_charge_types=''
    cleanup = "nphs_lps"
    allowed_bonds = "backbone"
    root = 'auto'
    outputfilename = None
    check_for_fragments = False
    bonds_to_inactivate = ""
    inactivate_all_torsions = False
    attach_nonbonded_fragments = False
    attach_singletons = False
    assign_unique_names = False    
    mode = 'automatic'
    dict = None

    for o, a in opt_list:
        if verbose: print('o=', o, ' a=', a)  # Python 3 print function
        if o in ('-l', '--l'):
            ligand_filename = os.path.basename(a)
            if verbose: print('set ligand_filename to ', a)
            ligand_ext = os.path.splitext(os.path.basename(a))[1]
        if o in ('-v', '--v'):
            verbose = True
            if verbose: print('set verbose to ', True)
        if o in ('-o', '--o'):
            outputfilename = a
            if verbose: print('set outputfilename to ', a)
        # ... [rest of the option handling remains the same]

    if not ligand_filename:
        print('prepare_ligand4: ligand filename must be specified.')
        usage()
        sys.exit()

    if attach_singletons:
        attach_nonbonded_fragments = True
        if verbose: print("using attach_singletons so attach_nonbonded_fragments also")
    
    mols = Read(ligand_filename)
    if verbose: print('read ', ligand_filename)
    mol = mols[0]
    if len(mols)>1:
        if verbose: 
            print("more than one molecule in file")
        ctr = 1
        for m in mols[1:]:
            ctr += 1
            if len(m.allAtoms)>len(mol.allAtoms):
                mol = m
                if verbose:
                    print("mol set to ", ctr, "th molecule with", len(mol.allAtoms), "atoms")

    coord_dict = {}
    for a in mol.allAtoms: 
        coord_dict[a] = a.coords

    if assign_unique_names:
        for at in mol.allAtoms:
            if mol.allAtoms.get(at.name) > 1:
                at.name = at.name + str(at._uniqIndex + 1)
        if verbose:
            print("renamed %d atoms" % (len(mol.allAtoms)))

    mol.buildBondsByDistance()
    if charges_to_add is not None:
        preserved = {}
        preserved_types = preserve_charge_types.split(',') 
        for t in preserved_types:
            if not len(t): continue
            try:
                ats = mol.allAtoms.get(lambda x: x.autodock_element==t)
                for a in ats:
                    if a.chargeSet is not None:
                        preserved[a] = [a.chargeSet, a.charge]
            except AttributeError:
                ats = mol.allAtoms.get(lambda x: x.element==t)
                for a in ats:
                    if a.chargeSet is not None:
                        preserved[a] = [a.chargeSet, a.charge]
            if verbose:
                print(" preserved = ")
                for key, val in preserved.items():
                    print("key=", key)
                    print("val =", val)

    if verbose:
        print("setting up LPO with mode=", mode)
        print("and outputfilename= ", outputfilename)
        print("and check_for_fragments=", check_for_fragments)
        print("and bonds_to_inactivate=", bonds_to_inactivate)

    LPO = AD4LigandPreparation(mol, mode, repairs, charges_to_add, 
                            cleanup, allowed_bonds, root, 
                            outputfilename=outputfilename,
                            dict=dict, check_for_fragments=check_for_fragments,
                            bonds_to_inactivate=bonds_to_inactivate, 
                            inactivate_all_torsions=inactivate_all_torsions,
                            attach_nonbonded_fragments=attach_nonbonded_fragments,
                            attach_singletons=attach_singletons)

    if charges_to_add is not None:
        for atom, chargeList in preserved.items():
            atom._charges[chargeList[0]] = chargeList[1]
            atom.chargeSet = chargeList[0]
            if verbose: print("set charge on ", atom.full_name(), " to ", atom.charge)

    if verbose: print("returning ", mol.returnCode)
    
    bad_list = []
    for a in mol.allAtoms:
        if a in coord_dict.keys() and a.coords != coord_dict[a]: 
            bad_list.append(a)
            
    if len(bad_list):
        print('%d atom coordinates changed!' % len(bad_list))
        for a in bad_list:
            print(a.name, ":", coord_dict[a], ' -> ', a.coords)
    else:
        if verbose: print("No change in atomic coordinates")
        
    if mol.returnCode != 0:
        sys.stderr.write(mol.returnMsg + "\n")
    sys.exit(mol.returnCode)