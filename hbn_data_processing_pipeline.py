#!/usr/bin/env python
# coding: utf-8

# # HBN Data Processing Pipeline
# 
# **Pipeline Overview:**
# 1. **Merge phenotype and diagnosis data** - Combine demographic info with clinical diagnoses
# 2. **Merge all data sources** - Combine NIH cognitive tests, Temporal Discounting tasks, and phenotype data
# 3. **Preprocess Temporal Discounting data** - Clean and engineer features from delay discounting measures
# 4. **Preprocess NIH Picture Sequence data** - Examine NIH Picture Sequence Memory test data
# 5. **Exploratory Data Analysis** - Build analysis-ready datasets and examine missingness
# 
# ---

# ## Part 1: Merge Phenotype and Diagnosis Data
# 
# Download phenotype data from multiple HBN releases and merge it with clinical diagnosis information.

# In[1]:


# Import all necessary libraries
from pathlib import Path
import os
import io
import re
import csv
import json
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set up folder structure
DATA = Path("data")
RAW = DATA / "raw"              # Raw downloaded files
INTERIM = DATA / "interim"      # Processed/merged files
PROCESSED = DATA / "processed"  # Final analysis-ready files
RESULTS = DATA / "results"      # Analysis results

# Create folders if they don't exist
for folder in [RAW, INTERIM, PROCESSED, RESULTS]:
    folder.mkdir(parents=True, exist_ok=True)

print("✓ Folder structure created")


# ### Helper Functions

# In[2]:


def http_text(url, timeout=60):
    """Download text content from a URL."""
    # Force HTTP for this specific server to avoid certificate issues
    if url.startswith("https://fcon_1000.projects.nitrc.org"):
        url = url.replace("https://", "http://", 1)
    
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def read_table_smart(url):
    """Download a CSV/TSV file and automatically detect the separator."""
    text = http_text(url)
    sample = text[:5000]
    
    try:
        # Try to automatically detect the delimiter
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[",", ";", "\t", "|"])
        separator = dialect.delimiter
    except Exception:
        # If automatic detection fails, use comma as default
        separators = [",", ";", "\t", "|"]
        separator = max(separators, key=sample.count)
    
    df = pd.read_csv(io.StringIO(text), sep=separator, engine="python")
    df.columns = [col.strip() for col in df.columns]  # Clean column names
    
    return df


def normalize_eid(value):
    """Clean and standardize participant EID (e.g., 'NDAR AA075 AMK' → 'NDARAA075AMK')."""
    if pd.isna(value):
        return np.nan
    
    cleaned = str(value).strip().upper()
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)  # Remove all non-alphanumeric
    
    return cleaned if cleaned else np.nan


def get_release_number(filename):
    """Extract release version number from filename (e.g., 'HBN_R10_Pheno.csv' → 10.0)."""
    match = re.search(r"_R(\d+)(?:_(\d+))?_Pheno\.csv$", filename)
    
    if not match:
        return 0.0
    
    major = int(match.group(1))
    minor = int(match.group(2)) if match.group(2) else 0
    
    return float(f"{major}.{minor}")

print("✓ Helper functions defined")


# ### Download Phenotype Files

# In[3]:


# Base URL for phenotype files
BASE = "http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/File/_pheno/"

# List of all phenotype files (Releases 1-11)
PHENO_FILES = [
    "HBN_R1_1_Pheno.csv",
    "HBN_R2_1_Pheno.csv",
    "HBN_R3_Pheno.csv",
    "HBN_R4_Pheno.csv",
    "HBN_R5_Pheno.csv",
    "HBN_R6_Pheno.csv",
    "HBN_R7_Pheno.csv",
    "HBN_R8_Pheno.csv",
    "HBN_R9_Pheno.csv",
    "HBN_R10_Pheno.csv",
    "HBN_R11_Pheno.csv"
]

pheno_frames = []

for filename in PHENO_FILES:
    url = BASE + filename
    
    try:
        print(f"Downloading {filename}...")
        df = read_table_smart(url)
        
        # Save to raw folder
        df.to_csv(RAW / filename, index=False)
        
        # Add tracking columns
        df["_release_file"] = filename
        df["_release_rank"] = get_release_number(filename)
        
        # Normalize participant ID
        if "EID" in df.columns:
            df["_EID"] = df["EID"].apply(normalize_eid)
        else:
            # Try to find an ID column
            id_column = next((col for col in df.columns 
                            if re.fullmatch(r"(participant_)?eid", col, flags=re.I)), None)
            if id_column:
                df["_EID"] = df[id_column].apply(normalize_eid)
            else:
                df["_EID"] = np.nan
        
        pheno_frames.append(df)
        print(f"  ✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
    except Exception as e:
        print(f"  ✗ WARNING: Failed to load {filename}: {e}")


# ### Combine and Deduplicate Phenotype Data
# 
# Combine all releases and keep only the latest data for each participant.

# In[4]:


# Combine all releases
pheno_all = pd.concat(pheno_frames, ignore_index=True)
print(f"Combined all releases: {pheno_all.shape[0]} rows, {pheno_all.shape[1]} columns")

# Remove rows without a valid EID
pheno_all = pheno_all[pheno_all["_EID"].notna()]

# Keep only the LATEST row for each participant
pheno_all_sorted = pheno_all.sort_values(["_EID", "_release_rank"])
pheno_latest = pheno_all_sorted.drop_duplicates("_EID", keep="last")

print(f"Latest data (one row per participant): {pheno_latest.shape[0]} participants")

# Save both versions
pheno_all.to_csv(RAW / "HBN_pheno_all_concat.csv", index=False)
pheno_latest.drop(columns=["_release_file", "_release_rank"]).to_csv(
    RAW / "HBN_pheno_latest.csv", index=False
)

print("\n✓ Phenotype data saved successfully")


# ### Load and Merge Diagnosis Data
# 
# Load the clinical diagnosis file and merge it with the phenotype data.

# In[5]:


# Load diagnosis file (must be in data/raw/ folder)
DIAG_FILE = "Diagnosis_ClinicianConsensus.csv"
diag_path = RAW / DIAG_FILE

diag = pd.read_csv(diag_path, low_memory=False)
print(f"✓ Loaded diagnosis file: {diag.shape[0]} rows, {diag.shape[1]} columns")


# In[6]:


# Create a set of known participant IDs for matching
known_eids = set(pheno_latest["_EID"].dropna().unique())
print(f"We have {len(known_eids)} known participant IDs from phenotype data")


def extract_eid_from_identifiers(identifier_value, known_ids):
    """Try to find a valid participant ID from the Identifiers field."""
    if pd.isna(identifier_value):
        return np.nan
    
    text = str(identifier_value).upper()
    tokens = re.split(r"[;,|\s]+", text)
    
    # Strategy 1: Look for exact matches to known IDs
    for token in tokens:
        cleaned_token = normalize_eid(token)
        if cleaned_token in known_ids:
            return cleaned_token
    
    # Strategy 2: Look for HBN-style IDs using pattern matching
    match = re.search(r"\bHBN[A-Z0-9]+\b", text)
    if match:
        cleaned_token = normalize_eid(match.group(0))
        if cleaned_token in known_ids:
            return cleaned_token
    
    return np.nan


# Extract EIDs from the diagnosis data
diag = diag.copy()
diag["_EID"] = diag["Identifiers"].apply(
    lambda val: extract_eid_from_identifiers(val, known_eids)
)

# Keep only rows where we successfully extracted an EID
diag_with_eid = diag[diag["_EID"].notna()]
diag_keyed = diag_with_eid.drop_duplicates("_EID")

print(f"✓ Successfully matched {diag_keyed.shape[0]} participants")
print(f"  (out of {diag.shape[0]} total diagnosis records)")


# In[7]:


# Merge phenotype and diagnosis data
merged = pheno_latest.merge(
    diag_keyed,
    on="_EID",
    how="inner",  # Keep only participants in BOTH datasets
    suffixes=("_pheno", "_dx")
)

print(f"✓ Merged dataset: {merged.shape[0]} participants, {merged.shape[1]} columns")

# Preview the merged data
base_cols = ["_EID", "Sex", "Age"]
dx_cols = [c for c in merged.columns if "DX_" in c][:8]
preview_cols = [c for c in (base_cols + dx_cols) if c in merged.columns]

print("\nPreview of merged data:")
merged[preview_cols].head(10)


# In[8]:


# Save merged phenotype + diagnosis data
merged_path = INTERIM / "HBN_pheno_with_diagnosis.csv"
merged.to_csv(merged_path, index=False)
print(f"✓ Saved merged dataset to: {merged_path}")


# ---
# 
# ## Part 2: Merge Master Data (NIH + Temporal Discounting + Phenotype)
# 
# Merge the NIH cognitive test data and Temporal Discounting task data with phenotype+diagnosis file.

# ### Load Raw Data Files
# 
# In `data/raw/` folder:
# - `NIH_final.csv`
# - `Temp_Disc_Final.csv`

# In[9]:


# Load the three main data sources
read_opts = dict(
    dtype=str, 
    keep_default_na=True, 
    na_values=["", "NA", "NaN"], 
    low_memory=False
)

nih = pd.read_csv(RAW / "NIH_final.csv", **read_opts)
td = pd.read_csv(RAW / "Temp_Disc_Final.csv", **read_opts)
ph = pd.read_csv(INTERIM / "HBN_pheno_with_diagnosis.csv", **read_opts)

print(f"NIH:        {nih.shape}")
print(f"Temp Disc:  {td.shape}")
print(f"Pheno+Dx:   {ph.shape}")


# ### Detect and Normalize EID Columns
# 
# Different files may have different column names for participant IDs. We'll standardize them.

# In[10]:


def detect_eid_col(df):
    """Find the participant ID column even if the header is unusual."""
    cols = list(df.columns)
    
    # Check if header contains 'EID' anywhere
    for c in cols:
        if "EID" in str(c).upper():
            return c
    
    # Look for values that look like NDAR IDs
    def looks_like_ndar(s):
        s = s.astype(str).str.upper()
        return s.str.startswith("NDAR").mean()
    
    best_c, best_score = None, 0.0
    for c in cols:
        try:
            score = looks_like_ndar(df[c])
            if score > best_score:
                best_c, best_score = c, score
        except Exception:
            pass
    
    if best_score >= 0.5:
        return best_c
    
    return None


def normalize_id(s):
    """Normalize participant IDs."""
    return (s.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.upper().str.strip()
            .str.replace(r"[^A-Z0-9]", "", regex=True))


# Detect and normalize EID columns for all three datasets
for name, df in [("NIH", nih), ("TempDisc", td), ("PhenoDx", ph)]:
    eid_col = detect_eid_col(df)
    if eid_col is None:
        raise KeyError(f"{name}: could not find an EID column")
    
    if eid_col != "EID":
        df.rename(columns={eid_col: "EID"}, inplace=True)
    
    df["_EID"] = normalize_id(df["EID"])
    print(f"{name}: detected EID column = '{eid_col}', unique IDs = {df['_EID'].nunique()}")


# ### Check Overlaps Between Datasets

# In[11]:


set_nih = set(nih["_EID"].dropna())
set_td = set(td["_EID"].dropna())
set_ph = set(ph["_EID"].dropna())

print("Overlap NIH ∩ TD:", len(set_nih & set_td))
print("Overlap NIH ∩ PhenoDx:", len(set_nih & set_ph))
print("Overlap TD  ∩ PhenoDx:", len(set_td & set_ph))
print("Overlap all three:", len(set_nih & set_td & set_ph))


# ### Merge All Three Datasets

# In[12]:


# Prepare datasets for merging (remove duplicate _EID columns)
td_left = td.loc[:, ["_EID"] + [c for c in td.columns if c not in ("EID", "_EID")]]
nih_right = nih.loc[:, ["_EID"] + [c for c in nih.columns if c not in ("EID", "_EID")]]

# Get demographics from phenotype data
ph_slim = ph.loc[:, ["_EID", "Sex", "Age"]].drop_duplicates("_EID")

# Perform the merge
master = (td_left.merge(nih_right, on="_EID", how="inner", suffixes=("_td", "_nih"))
                 .merge(ph_slim, on="_EID", how="inner"))

print(f"✓ Master dataset shape: {master.shape}")

# Save the merged master file
out_csv = INTERIM / "NIH_TempDisc_pheno_diagnosis.csv"
master.to_csv(out_csv, index=False)
print(f"✓ Saved: {out_csv}")


# ### Check Missingness in Merged Data

# In[13]:


# Calculate column-wise missing percentages
missing_pct = master.isna().mean()

# Drop columns that are 100% missing
drop_cols = missing_pct[missing_pct == 1.0].index.tolist()
master_nz = master.drop(columns=drop_cols)
print(f"Dropped {len(drop_cols)} columns with 100% missing.")

# Recompute missingness
miss = master_nz.isna().mean().sort_values(ascending=False)
miss_df = pd.DataFrame({"column": miss.index, "missing_pct": miss.values})

# Create 10% bins for missingness
bins = np.arange(0, 1.01, 0.1)
labels = [f"{int(b*100)}–{int((b+0.1)*100)}%" for b in bins[:-1]]
miss_df["bin"] = pd.cut(
    miss_df["missing_pct"], 
    bins=bins, 
    labels=labels, 
    include_lowest=True, 
    right=False
)

# Show top 20 most-missing columns
print("\nTop 20 most-missing columns:")
miss_df.assign(missing_pct=lambda d: (d["missing_pct"]*100).round(1)).head(20)


# ---
# 
# ## Part 3: Preprocess Temporal Discounting Data
# 
# Clean the Temporal Discounting (delay discounting) data and create analysis-ready features.

# In[14]:


# Load the merged master file
MASTER = INTERIM / "NIH_TempDisc_pheno_diagnosis.csv"
df = pd.read_csv(MASTER, low_memory=False)
print(f"Loaded: {df.shape}")


# ### Find Temporal Discounting Columns
# 
# The column names have long prefixes, so find them by their suffixes.

# In[15]:


def find_col(df, tail):
    """Find a column by its ending (suffix)."""
    tail_low = tail.lower()
    cands = [c for c in df.columns
             if c.lower().endswith(tail_low) or c.split(",")[-1].strip().lower() == tail_low]
    print(f"{tail}: {cands[:3] if len(cands) > 3 else cands}")
    return cands[0] if cands else None


# Find the key TD columns
k1 = find_col(df, "Temp_Disc_run1_k")
k2 = find_col(df, "Temp_Disc_run2_k")
ed1 = find_col(df, "Temp_Disc_run1_ed50")
ed2 = find_col(df, "Temp_Disc_run2_ed50")

key_cols = [c for c in [k1, k2, ed1, ed2] if c is not None]
assert len(key_cols) > 0, "No TD columns detected—check the file headers."


# ### Examine Missingness in TD Data

# In[16]:


# Coerce to numeric
df[key_cols] = df[key_cols].apply(pd.to_numeric, errors="coerce")

# Calculate missingness
miss_tbl = (
    df[key_cols].isna()
    .agg(["sum", "mean"]).T
    .rename(columns={"sum": "n_missing", "mean": "missing_pct"})
    .assign(
        n_rows=len(df),
        n_present=lambda x: x.n_rows - x.n_missing
    )
    .sort_values("missing_pct", ascending=False)
)

print("Missingness in TD columns:")
miss_tbl


# In[17]:


# Visualize missingness
(miss_tbl["missing_pct"] * 100).sort_values().plot(kind="barh", figsize=(6, 3))
plt.xlabel("% missing")
plt.ylabel("TD column")
plt.title("TD Missingness")
plt.tight_layout()
plt.show()


# ### Clean Column Names and Create Features

# In[18]:


# Create short, clean names
rename_map = {}
if k1:  rename_map[k1] = "k1"
if k2:  rename_map[k2] = "k2"
if ed1: rename_map[ed1] = "ed50_1"
if ed2: rename_map[ed2] = "ed50_2"

df = df.rename(columns=rename_map)


def to_num(s):
    """Convert column to numeric, handling commas and spaces."""
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "NA": np.nan, "null": np.nan}),
        errors="coerce"
    )


# Convert to numeric
for c in ["k1", "k2", "ed50_1", "ed50_2"]:
    if c in df.columns:
        df[c] = to_num(df[c])

print("First few rows:")
df[["k1", "k2", "ed50_1", "ed50_2"]].head()


# ### Validate Data: Check Relationship Between k and ED50
# 
# In delay discounting theory, ED50 ≈ 1/k (both in days). Let's verify this.

# In[19]:


# Check: (1/k) should be close to ed50
for kcol, dcol in [("k1", "ed50_1"), ("k2", "ed50_2")]:
    if kcol in df.columns and dcol in df.columns:
        ok = df[kcol].gt(0) & df[dcol].notna()
        ratio = (1.0 / df.loc[ok, kcol]) / df.loc[ok, dcol]
        print(f"Median((1/{kcol})/{dcol}) = {np.nanmedian(ratio):.10f}")
        # Should be very close to 1.0 if data is valid


# ### Create Analysis Features

# In[20]:


# Create derived features
if "k1" in df and "k2" in df:
    df["k_mean"] = df[["k1", "k2"]].mean(axis=1, skipna=True)
    df["logk_mean"] = np.log(df["k_mean"].clip(lower=1e-6))

if "ed50_1" in df and "ed50_2" in df:
    df["ed50_mean"] = df[["ed50_1", "ed50_2"]].mean(axis=1, skipna=True)

df["k_abs_diff"] = (df.get("k1", np.nan) - df.get("k2", np.nan)).abs()

# Display the new features
feat_cols = ["k1", "k2", "k_mean", "logk_mean", "ed50_1", "ed50_2", "ed50_mean", "k_abs_diff"]
feat_cols = [c for c in feat_cols if c in df.columns]

print("Analysis features:")
df[feat_cols].head()


# ### Save Processed Data with TD Features

# In[21]:


# Save back to interim folder with TD features included
OUT = INTERIM / "NIH_TempDisc_pheno_diagnosis_withTD.csv"
df.to_csv(OUT, index=False)
print(f"✓ Saved: {OUT}")


# ---
# 
# ## Part 4: Preprocess NIH Picture Sequence Memory Data
# 
# Examine the NIH Picture Sequence Memory test data.

# In[22]:


# Load the master file with TD features
df = pd.read_csv(INTERIM / "NIH_TempDisc_pheno_diagnosis_withTD.csv", low_memory=False)
print(f"Loaded: {df.shape}")


# ### Identify Picture Sequence Memory Columns

# In[23]:


# Find all Picture Sequence columns
prefix = "NIH_final,NIH_Picture_Seq_"
ps_cols = [c for c in df.columns if c.startswith(prefix)]

print(f"Found {len(ps_cols)} Picture Sequence columns:")
for c in ps_cols:
    print(" •", c)


# ### Check Missingness in Picture Sequence Data

# In[24]:


# Per-column missingness
col_missing = (
    df[ps_cols].isna()
    .agg(["sum", "mean"])
    .T.rename(columns={"sum": "n_missing", "mean": "missing_pct"})
)
col_missing["n_rows"] = len(df)
col_missing["n_present"] = col_missing["n_rows"] - col_missing["n_missing"]
col_missing = col_missing.sort_values("missing_pct", ascending=False)

print("Missingness in Picture Sequence columns:")
col_missing


# In[25]:


# Visualize missingness
ax = (col_missing["missing_pct"] * 100).sort_values().plot(kind="barh", figsize=(7, 4))
ax.set_xlabel("% missing")
ax.set_ylabel("Picture Seq column")
plt.tight_layout()
plt.show()


# ### Per-Participant Completeness
# 
# How many participants have complete vs. missing Picture Sequence data?

# In[26]:


# Calculate fraction of missing PS fields per participant
row_missing_frac = df[ps_cols].isna().mean(axis=1)
row_bins = pd.cut(
    row_missing_frac, 
    bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
    right=False
)
row_summary = row_bins.value_counts().sort_index().rename("n_participants").to_frame()
row_summary["pct_participants"] = row_summary["n_participants"] / len(df)

print("Per-participant completeness:")
row_summary


# In[36]:


# List participants with ALL PS fields missing
eid_col = next((c for c in df.columns if c.endswith("_EID") or c == "_EID" or c == "EID"), None)

if eid_col:
    all_missing_idx = row_missing_frac.eq(1.0)
    print(f"Participants with ALL PS fields missing: {int(all_missing_idx.sum())}")


# #### Picture Sequence Data currently not usable

# ---
# 
# ## Part 5: Master Sheet Preprocessing

# In[28]:


# Load the fully processed data
df = pd.read_csv(INTERIM / "NIH_TempDisc_pheno_diagnosis_withTD.csv", low_memory=False)
print(f"Loaded: {df.shape}")


# ### Drop Columns with ≥90% Missing

# In[29]:


# Calculate missingness
col_missing = df.isna().mean()

# Identify columns to drop
hard_drop = col_missing[col_missing >= 0.90].index.tolist()
df1 = df.drop(columns=hard_drop)

print(f"Dropped {len(hard_drop)} columns with ≥90% missing")
print(f"Remaining shape: {df1.shape}")


# ### Build Feature Tiers
# 
# Organize features by missingness level:
# - **Core features**: ≤20% missing
# - **Extended features**: 20-40% missing

# In[30]:


# Define feature tiers
core_cols = col_missing[(col_missing <= 0.20)].index.tolist()
extended_cols = col_missing[(col_missing > 0.20) & (col_missing <= 0.40)].index.tolist()

print(f"Core features (≤20% missing): {len(core_cols)}")
print(f"Extended features (20-40% missing): {len(extended_cols)}")


# ### Select Working Feature Set
# 
# Pick must-have variables and relevant NIH cognitive test measures.

# In[31]:


# Must-have variables
must_have = [
    "_EID", "Age", "Sex",
    "logk_mean", "ed50_mean",
    "k_mean", "k_abs_diff"
]

# Select relevant NIH cognitive measures
nih_candidates = [
    c for c in core_cols 
    if c.startswith("NIH_final,NIH_") and 
    any(k in c for k in ["Flanker", "Processing", "List_Sort"])
]

feat_core = sorted(set(must_have + nih_candidates))
print(f"Working feature set: {len(feat_core)} columns")


# ### Create Complete-Case Views
# 
# Create datasets with minimal missingness for analysis.

# In[32]:


def complete_case_view(cols, min_row_frac=0.9):
    """
    Keep rows where at least min_row_frac of the columns are non-missing.
    """
    sub = df1[cols].copy()
    req = [c for c in cols if c != "_EID"]
    keep = sub[req].notna().mean(axis=1) >= min_row_frac
    return df1.loc[keep, cols]


# Core view: keep rows with ≥90% completeness
core_view = complete_case_view(feat_core, min_row_frac=0.9)
print(f"Core view shape: {core_view.shape}")

# Check missingness in the core view
miss_report = (
    core_view[feat_core].isna().mean()
    .mul(100)
    .round(1)
    .sort_values(ascending=False)
)
print("\nMissingness in core view:")
print(miss_report.head(10))


# ### Create Extended View
# 
# Include more features with slightly more missingness.

# In[33]:


# Add extended features
feat_ext = sorted(
    set(feat_core + [
        c for c in extended_cols 
        if c.startswith("NIH_final,NIH_") and 
        any(k in c for k in ["Flanker", "Processing", "List_Sort"])
    ])
)

ext_view = complete_case_view(feat_ext, min_row_frac=0.85)
print(f"Extended view shape: {ext_view.shape}")


# ### Save Analysis-Ready Datasets

# In[34]:


# Save versioned files
tag = "v1"
core_path = PROCESSED / f"hbn_core_view_{tag}.csv"
ext_path = PROCESSED / f"hbn_extended_view_{tag}.csv"

core_view.to_csv(core_path, index=False)
ext_view.to_csv(ext_path, index=False)

print(f"✓ Saved core view: {core_path}")
print(f"✓ Saved extended view: {ext_path}")


# ### Summary Statistics

# In[35]:


# Basic descriptive statistics
summary_cols = ["Age", "Sex", "logk_mean", "ed50_mean", "k_mean"]
summary_cols = [c for c in summary_cols if c in core_view.columns]

print("Summary statistics (core view):")
core_view[summary_cols].describe()


# ---
# 
# ## Summary of outputs:
# - `data/interim/HBN_pheno_with_diagnosis.csv` - Phenotype + diagnosis data
# - `data/interim/NIH_TempDisc_pheno_diagnosis.csv` - All data merged
# - `data/interim/NIH_TempDisc_pheno_diagnosis_withTD.csv` - Add TD features
# - `data/processed/hbn_core_view_v1.csv` - Analysis-ready core dataset
# - `data/processed/hbn_extended_view_v1.csv` - Analysis-ready extended dataset
# 
# **Next steps:**
# - Use the core/extended views for statistical analyses
# - Build predictive models
# - Create visualizations
# - Run group comparisons

# In[ ]:




