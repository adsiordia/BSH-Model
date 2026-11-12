# Free amine nucleophiles seen as conjugates in Trimmed_remove_proteomics (4).csv.
# Keys match the amine token in the CSV column names; LABEL fixes names where the
# CSV token is a misnomer.
SMILES = {
 "alanine":                  "C[C@@H](N)C(O)=O",
 "asparagine":               "NC(=O)C[C@H](N)C(O)=O",
 "methionine":               "CSCC[C@H](N)C(O)=O",
 "threonine":                "C[C@@H](O)[C@H](N)C(O)=O",
 "serine":                   "OC[C@H](N)C(O)=O",
 "phenylalanine":            "N[C@@H](Cc1ccccc1)C(O)=O",
 "glutamine":                "NC(=O)CC[C@H](N)C(O)=O",
 "histidine":                "N[C@@H](Cc1c[nH]cn1)C(O)=O",
 "arginine":                 "N[C@@H](CCCNC(N)=N)C(O)=O",
 "lysine":                   "NCCCC[C@H](N)C(O)=O",
 "ornithine":                "NCCC[C@H](N)C(O)=O",
 "citrulline":               "NC(=O)NCCC[C@H](N)C(O)=O",
 "proline":                  "OC(=O)[C@@H]1CCCN1",
 "cystine":                  "N[C@@H](CS)C(O)=O",   # free cysteine, not the disulfide
 "2,3_diaminopropionic acid":"NC[C@H](N)C(O)=O",
 "GABA":                     "NCCCC(O)=O",
 "glyglycine":               "NCC(=O)NCC(O)=O",
 "dopamine":                 "NCCc1ccc(O)c(O)c1",
 "tyramine":                 "NCCc1ccc(O)cc1",
 "3_methoxytyramine":        "NCCc1ccc(O)c(OC)c1",
 "serotonin":                "NCCc1c[nH]c2ccc(O)cc12",
 "tryptamine":               "NCCc1c[nH]c2ccccc12",
 "2_aminophenol":            "Nc1ccccc1O",
 "putrescine":               "NCCCCN",
 "cadaverine":               "NCCCCCN",
}
LABEL = {"cystine": "cysteine", "2,3_diaminopropionic acid": "Dap", "glyglycine": "glycylglycine"}
def label(k): return LABEL.get(k, k)
