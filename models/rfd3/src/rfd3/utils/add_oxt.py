from biotite.structure import Atom, AtomArray
import numpy as np

def add_oxt_each_chain(atom_array):
    """
    For each chain in the AtomArray, check and add OXT atom if missing
    at the C-terminal residue.

    Args:
        atom_array (AtomArray): The input AtomArray.

    Returns:
        AtomArray: The updated AtomArray with OXT atoms added to chains as needed.
    """
    updated_chains = []
    chain_ids = np.unique(atom_array.chain_id)
    
    for chain_id in chain_ids:
        chain_mask = atom_array.chain_id == chain_id
        chain_array = atom_array[chain_mask]
        updated_chain = add_oxt_to_chain(chain_array)
        updated_chains.append(updated_chain)
    
    # Concatenate all processed chains back into a single AtomArray
    return AtomArray.concatenate(updated_chains)


def add_oxt_to_chain(chain_array):
    """
    Process a single chain's AtomArray to add the OXT atom at the C-terminal residue.

    Args:
        chain_array (AtomArray): The AtomArray for a single chain.

    Returns:
        AtomArray: Updated AtomArray for the chain with OXT added if applicable.
    """
    # Identify the C-terminal residue
    c_terminal_res_id = np.max(chain_array.res_id)
    c_terminal_mask = chain_array.res_id == c_terminal_res_id
    c_terminal_atoms = chain_array[c_terminal_mask]
    
    # Check if OXT already exists
    if "OXT" in c_terminal_atoms.atom_name.tolist():
        return chain_array  # Return the original chain if OXT exists
    
    # Extract coordinates for O, CA, C
    try:
        c_coord = c_terminal_atoms.coord[c_terminal_atoms.atom_name == "C"][0]
        ca_coord = c_terminal_atoms.coord[c_terminal_atoms.atom_name == "CA"][0]
        o_coord = c_terminal_atoms.coord[c_terminal_atoms.atom_name == "O"][0]
    except IndexError:
        return chain_array  # Skip this chain if critical atoms are missing
    
    # Calculate OXT coordinates
    oxt_coord = calculate_oxt_coord(o_coord, ca_coord, c_coord)
    
    # Add OXT atom
    oxt_atom = Atom(
        coord=oxt_coord,
        atom_name="OXT",
        res_id=c_terminal_res_id,
        chain_id=chain_array.chain_id[0],  # Use the chain ID from the chain
        element="O",
    )
    
    # Append OXT atom to the chain
    return AtomArray.concatenate([chain_array, oxt_atom])


def calculate_oxt_coord(o_coord, ca_coord, c_coord):
    """
    Calculate the coordinates for the OXT atom based on the rotation of the O atom
    around the C-CA vector by 180 degrees.

    Args:
        o_coord (numpy.ndarray): The coordinates of the O atom.
        ca_coord (numpy.ndarray): The coordinates of the CA atom.
        c_coord (numpy.ndarray): The coordinates of the C atom.

    Returns:
        numpy.ndarray: The calculated coordinates of the OXT atom.
    """
    # Calculate the vector from C to CA and O
    c_ca_vector = ca_coord - c_coord
    o_c_vector = o_coord - c_coord
    
    # Reflect O around the C-CA vector to get OXT
    oxt_vector = o_c_vector - 2 * np.dot(o_c_vector, c_ca_vector) / np.linalg.norm(c_ca_vector)**2 * c_ca_vector
    oxt_coord = c_coord + oxt_vector
    
    return oxt_coord
