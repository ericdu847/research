import subprocess, sys
import os
import urllib.request

# Step 1: Ensure dependencies are installed in the script environment
print("Installing biopython, pandas, and matplotlib...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'biopython', 'pandas', 'matplotlib'], check=True)

# Step 2: Set global variables
TM1_START = 32  
pdb_path = 'gpr151_pipeline/gpr151_alphafold.pdb'
AF_URL = 'https://alphafold.ebi.ac.uk/files/AF-Q8TDV0-F1-model_v6.pdb'

# Step 3: Imports & Ticker tools
from Bio import PDB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator  # <-- For the advanced tick marks
from Bio.Data.PDBData import protein_letters_3to1_extended

def three_to_one(three_letter_code):
    return protein_letters_3to1_extended.get(three_letter_code.upper(), 'X')

kd_scale = {
    'A': 1.8,  'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8,  'K': -3.9, 'M': 1.9,  'F': 2.8,  'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Auto-Download PDB
os.makedirs('gpr151_pipeline', exist_ok=True)
if not os.path.exists(pdb_path):
    print(f"📦 Fetching human GPR151 from AlphaFold DB...")
    urllib.request.urlretrieve(AF_URL, pdb_path)

# Step 4: Extract structural metrics
parser = PDB.PDBParser(QUIET=True)
structure = parser.get_structure('gpr151', pdb_path)

data = []
for model in structure:
    for chain in model:
        for residue in chain:
            res_name = residue.get_resname()
            res_id = residue.get_id()[1]
            ca_atoms = [a for a in residue if a.get_name() == 'CA']
            
            if ca_atoms:
                try:
                    aa = three_to_one(res_name)
                except KeyError:
                    aa = 'X'
                
                plddt = ca_atoms[0].get_bfactor()  
                hydropathy = kd_scale.get(aa, 0.0)
                
                data.append({
                    'residue_id': res_id,
                    'aa': aa,
                    'plddt': plddt,
                    'hydropathy': hydropathy
                })

df = pd.DataFrame(data)

# Step 5: Smooth hydropathy
WINDOW_SIZE = 9
df['hydropathy_smooth'] = df['hydropathy'].rolling(window=WINDOW_SIZE, center=True, min_periods=1).mean()

# =========================================================================
# Step 6: Construct the Large-Scale Master Plot
# =========================================================================
# Expanded the width significantly (24x9 inches) to accommodate fine tick intervals cleanly
fig, ax1 = plt.subplots(figsize=(24, 9))

# --- AXIS 1: pLDDT (Left Y-Axis) ---
ax1.plot(df['residue_id'], df['plddt'], color='steelblue', linewidth=1.5, label='pLDDT Confidence')
ax1.set_ylabel('pLDDT Confidence Score', color='steelblue', fontweight='bold', fontsize=12)
ax1.tick_params(axis='y', labelcolor='steelblue', labelsize=10)
ax1.set_ylim(-15, 105)  

# Background structural zones
ax1.axhspan(0, 70, alpha=0.12, color='red', zorder=0)
ax1.axhspan(70, 90, alpha=0.04, color='yellow', zorder=0)
ax1.axhspan(90, 100, alpha=0.04, color='green', zorder=0)

# --- AXIS 2: Hydropathy (Right Y-Axis) ---
ax2 = ax1.twinx()
ax2.plot(df['residue_id'], df['hydropathy_smooth'], 
         color='darkorange', linewidth=1.8, linestyle='-', label=f'Hydropathy (Window={WINDOW_SIZE})')
ax2.set_ylabel('Kyte-Doolittle Hydropathy Index', color='darkorange', fontweight='bold', fontsize=12)
ax2.tick_params(axis='y', labelcolor='darkorange', labelsize=10)
ax2.set_ylim(-5, 5)
ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)

# Truncation Boundary Callout
ax1.axvline(x=TM1_START, color='crimson', linestyle='--', linewidth=2.5, 
            label=f'Truncation Site (Residue {TM1_START})', zorder=4)

# --- ADVANCED TICK MARK & GRID CONTROL ENGINE ---
# Set Major Ticks every 10 residues, Minor Ticks every 2 residues
ax1.xaxis.set_major_locator(MultipleLocator(10))
ax1.xaxis.set_minor_locator(MultipleLocator(2))

# Style the ticks: Long prominent lines for majors, short subtle stubs for minors
ax1.tick_params(axis='x', which='major', colors='black', labelsize=9, rotation=45, length=8, width=1.5)
ax1.tick_params(axis='x', which='minor', colors='dimgray', length=4, width=1.0)

# Add matching structural grid lines map matching the major ticks
ax1.grid(visible=True, which='major', axis='x', color='gainsboro', linestyle='-', linewidth=0.7, alpha=0.7)
ax1.grid(visible=True, which='minor', axis='x', color='whitesmoke', linestyle=':', linewidth=0.5, alpha=0.5)

# --- Primary Structure Track (Bottom Layer) ---
# Expanded horizontal sizing allows us to slightly bump the sequence text sizing safely
for _, row in df.iterrows():
    aa_color = 'firebrick' if row['hydropathy'] > 0 else 'navy'
    ax1.text(row['residue_id'], -7.5, row['aa'], fontsize=5.5, 
             ha='center', va='center', fontfamily='monospace', 
             color=aa_color, fontweight='bold', alpha=0.85)

ax1.text(df['residue_id'].min() - 3, -7.5, 'Seq:', fontsize=10, 
         ha='right', va='center', fontfamily='sans-serif', fontweight='bold')

# --- Formatting & Save ---
ax1.set_xlabel('Residue Position Number', fontweight='bold', fontsize=12, labelpad=15)
ax1.set_xlim(df['residue_id'].min() - 1, df['residue_id'].max() + 1)

# Handled the combined legend cleanly (Fixed the typo from earlier version)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.95, fontsize=10)

plt.title('GPR151 High-Resolution Dashboard: Structural Confidence vs Hydropathy Profile', 
          fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()

# Save at 300 DPI for publication-ready zooming capabilities
plt.savefig('gpr151_pipeline/unified_sequence_dashboard_hd.png', dpi=300)
plt.show()