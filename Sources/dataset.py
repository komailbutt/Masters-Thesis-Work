# Sources/dataset.py

import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import skrf as rf
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from tqdm import tqdm

# Datatset and Dataloader are core abstractions in PyTorch that decouple how you define your data from how you efficiently iterate over it in training loops.

# Datatset Class is essentially a blueprint. When you create a custom Dataset, you decide how data is loaded and returned.
# It defines three things: 1) __init__(), 2) __len__(), 3)__getitem__(index)
class PDNDataset(Dataset):
   """
   Loads Z11 from .s4p files.

   Returns:
       x   : Tensor [C, Nf] depending on repr_name
             repr_name="ln_phase" -> C=2 : (mag[ln(|Z|)], phase[rad])
             repr_name="re_im"    -> C=2 : (Re(Z11), Im(Z11))
             repr_name="complex"  -> C=1 : complex Z11 as complex64  (NEW)
       geo : Tensor [P] geometry params (RAW, NOT normalized in dataset)
       sid : simulation index

   Also stores:
       self.freq_hz : torch.Tensor [Nf] (float64) (from cache or parsing)
   """

# -------------------------------
# 1) __init__() tells how data should be loaded.
# -------------------------------
   def __init__(
       self,
       data_dir: str,
       csv_file: str,
       subset_size: Optional[int] = None,
       repr_name: str = "ln_phase",   # representation tag (for cache naming)

       # (NEW) subset selection
       subset_sampling: str = "first",  # "first" or "random"
       subset_seed: int = 42,           # used only when subset_sampling="random"

       # (NEW) geometry normalization
       # IMPORTANT:
       # We keep these args for backward compatibility with the notebook call signature,
       # but we DO NOT normalize geometry inside the Dataset anymore.
       # Geometry scaling belongs to Stage-II (train-only scaler) to avoid leakage.
       geo_norm: str = "none",   # kept for compatibility; ignored internally (raw geo returned)
       geo_scaler_path: Optional[str] = None,  # kept for compatibility; ignored internally
   ):
       self.data_dir = data_dir
       self.csv_file = csv_file
       self.subset_size = subset_size
       self.repr_name = repr_name

       self.subset_sampling = subset_sampling
       self.subset_seed = int(subset_seed)

       # kept for compatibility (do NOT use in dataset)
       self.geo_norm = geo_norm
       self.geo_scaler_path = geo_scaler_path

       # -------------------------------
       # Cache naming (repr-aware)
       # -------------------------------
       subset_tag = "FULL" if subset_size is None else f"N{subset_size}"

       # (NEW) include sampling strategy + seed so cache never lies
       sampling_tag = f"{subset_sampling}"
       seed_tag = f"seed{self.subset_seed}" if subset_sampling == "random" else "seedNA"

       # (NEW) keep geo tag but force it to RAW (we never normalize inside dataset)
       # This prevents accidental mixing with old caches.
       geo_tag = "geo_raw"

       cache_name = f"data_cache_{repr_name}_{subset_tag}_{sampling_tag}_{seed_tag}_{geo_tag}.pt"
       self.cache_path = os.path.join(os.path.dirname(csv_file), cache_name)

       # -------------------------------
       # Load CSV
       # -------------------------------
       df_full = pd.read_csv(csv_file)
       if "simu_index" not in df_full.columns:
           raise ValueError("CSV must contain 'simu_index' column")

       df_full = df_full.copy()

       # Sorting is mandatory for stage-II mapping (ANN)
       geo_cols = [c for c in df_full.columns if c != "simu_index"]  # All columns except simu_index are geometry

       # -------------------------------
       # Subset selection (NEW: random or first-N)
       # -------------------------------
       if subset_size is not None:
           if subset_sampling == "first":
               print(f"Limit active: Using first {subset_size} samples.")
               df = df_full.sort_values("simu_index").reset_index(drop=True)
               df = df.iloc[:subset_size].copy()   # copy() prevents pandas views vs copies issues

           elif subset_sampling == "random":
               print(f"Limit active: Using RANDOM {subset_size} samples (seed={self.subset_seed}).")
               df = df_full.sample(n=subset_size, random_state=self.subset_seed).copy()

               # Sorting is mandatory for stage-II mapping (ANN)
               df = df.sort_values("simu_index").reset_index(drop=True)

           else:
               raise ValueError("subset_sampling must be 'first' or 'random'")
       else:
           df = df_full.sort_values("simu_index").reset_index(drop=True)

       self.params_df = df

       # (NEW) Keep subset id list (important for logging/reproducibility)
       self.selected_sids = df["simu_index"].astype(int).to_numpy()

       # -------------------------------
       # Load cache OR parse S4P
       # -------------------------------
       self.freq_hz = None  # cached frequency grid (float64) for integrity + plotting

       if os.path.exists(self.cache_path):
           print(f"Loading cached dataset: {self.cache_path}")
           cached = torch.load(self.cache_path, map_location="cpu")  # map_location="cpu" ensures it loads even if cache was made on GPU

           # Loading already processed tensors
           self.impedance_data = cached["impedance"]
           self.geo_params = cached["geometry"]
           self.simu_ids = cached["simu_ids"]

           # NEW: load cached frequency grid if present
           freq_np = cached.get("freq_hz", None)
           if freq_np is not None:
               self.freq_hz = torch.from_numpy(np.asarray(freq_np, dtype=np.float64))

           # (NEW) load cache metadata (safe if missing in old caches)
           self.cache_meta = cached.get("meta", {})

           if self.freq_hz is None:
               print("WARNING: cache has no 'freq_hz' (old cache). Consider regenerating cache.")

       else:
           print(f"Loading S4P files (repr='{repr_name}') ...")

           # Calls static method that loops over files and returns numpy arrays
           imp_np, geo_np, sid_np, freq_np, skipped, failed_ids = self._process_files(df, data_dir, repr_name)

           # FAST tensor creation
           t = torch.from_numpy(imp_np)

           # NEW: preserve complex dtype if repr="complex"
           self.impedance_data = t if torch.is_complex(t) else t.float()

           self.geo_params = torch.from_numpy(geo_np).float()
           self.simu_ids = torch.from_numpy(sid_np).long()

           # NEW: store frequency grid (double for exactness)
           self.freq_hz = torch.from_numpy(np.asarray(freq_np, dtype=np.float64))

           # (NEW) cache metadata for reproducibility/debugging
           meta = {
               "repr_name": repr_name,
               "subset_size": subset_size,
               "subset_sampling": subset_sampling,
               "subset_seed": int(self.subset_seed),

               # IMPORTANT: dataset returns RAW geo only
               "geo_norm": "raw_in_dataset",
               "geo_scaler_path": None,

               "num_loaded": int(self.impedance_data.shape[0]),
               "num_skipped": int(skipped),
               "failed_ids_first50": failed_ids,

               "csv_file": os.path.abspath(csv_file),
               "data_dir": os.path.abspath(data_dir),
               "subset_ids_first50": self.selected_sids[:50].tolist(),
               "subset_ids_count": int(len(self.selected_sids)),

               # NEW: freq metadata
               "n_freq": int(self.freq_hz.numel()),
               "f_start_hz": float(self.freq_hz[0].item()),
               "f_stop_hz": float(self.freq_hz[-1].item()),
           }
           self.cache_meta = meta

           torch.save(
               {
                   "impedance": self.impedance_data,
                   "geometry": self.geo_params,
                   "simu_ids": self.simu_ids,
                   "freq_hz": self.freq_hz.cpu().numpy(),
                   "meta": meta,
               },
               self.cache_path,
           )
           print(f"Cached dataset saved to: {self.cache_path}")
           print(f"Cache meta: loaded={meta['num_loaded']}, skipped={meta['num_skipped']}")
           if meta["num_skipped"] > 0:
               print(f"WARNING: skipped {meta['num_skipped']} files. First failed IDs: {meta['failed_ids_first50']}")

   # static method= only uses what you pass into it. Takes df and data_dir and returns numpy arrays.
   @staticmethod
   def _process_files(
       df: pd.DataFrame,
       data_dir: str,
       repr_name: str,
   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, list]:
   # Tuple[..] this func returns multiple objects in fixed structure
   # UPDATED: now also returns freq_hz and failed_ids list

       impedance_list = []
       geometry_list = []
       sid_list = []
       skipped = 0
       failed_ids = []

       freq_hz = None

       geo_cols = [c for c in df.columns if c != "simu_index"] # All columns except simu_index are geometry

       for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing Files"): # iterrows gives each row containing simu_index +geo params
           sim_id = int(row["simu_index"])
           fpath = os.path.join(data_dir, f"simu_{sim_id}.s4p") # If csv has simu_index=100000, it loads, simu_100000.s4p

           try:
               ntwk = rf.Network(fpath)  # ntwk.z shape is [Nf,4,4]

               # NEW: store + validate frequency grid (integrity check)
               if freq_hz is None:
                   freq_hz = ntwk.f.copy()
               else:
                   if not np.allclose(ntwk.f, freq_hz, rtol=1e-10, atol=1e-12):
                       raise ValueError(f"Frequency grid mismatch in simu_{sim_id}.s4p")

               z11 = ntwk.z[:, 0, 0]     # z11 is 0,0 across all frequencies

               # representation switch
               if repr_name == "ln_phase":
                   # --- magnitude in ln(|Z|) ---
                   mag_ohm = np.abs(z11)
                   mag_ln = np.log(mag_ohm + 1e-12)     # ln(|Z|) compresses range, 1e-12 avoids log(0)

                   # --- phase in radians ---
                   phase = np.angle(z11)

                   feat = np.stack([mag_ln, phase], axis=0).astype(np.float32)

               elif repr_name == "re_im":
                   # --- real/imag ---
                   re = np.real(z11).astype(np.float32)
                   im = np.imag(z11).astype(np.float32)
                   feat = np.stack([re, im], axis=0).astype(np.float32)

               elif repr_name == "complex":
                   # --- true complex representation ---
                   # final per-sample shape = [1, Nf] complex64
                   feat = z11.astype(np.complex64)[None, :]

               else:
                   raise ValueError(f"Unknown repr_name='{repr_name}'. Use 'ln_phase' or 're_im' or 'complex'.")

               impedance_list.append(feat)  # final per-sample shape depends on repr_name
               geometry_list.append(row[geo_cols].to_numpy(np.float32))
               sid_list.append(sim_id) # ensures alignment

           except Exception:
               skipped += 1
               if len(failed_ids) < 50:
                   failed_ids.append(sim_id)
               continue

       if len(impedance_list) == 0:
           raise RuntimeError("No valid S4P files loaded.") # Fail early if nothing loaded.

       if freq_hz is None:
           raise RuntimeError("Frequency grid could not be established (no valid files loaded).")

       return (
           np.stack(impedance_list),                 # impedance: [N, C, Nf]
           np.stack(geometry_list),                  # geometry: [N, P]
           np.asarray(sid_list, dtype=np.int64),     # sid: [N]
           np.asarray(freq_hz, dtype=np.float64),    # freq: [Nf]
           skipped,                                  # number of skipped files
           failed_ids,                               # record first failed IDs
       )

# -------------------------------
# 2) __len__() it returns the total number of samples.
# -------------------------------
   def __len__(self):
       return self.impedance_data.shape[0] # [0] is dataset length.

# -------------------------------
# 3) __getitem__(index) it returns the data (and label) at the given index.
# -------------------------------
   def __getitem__(self, idx):
       x = self.impedance_data[idx].clone()  # Clone because later we normalize batches; avoids modifying stored tensors.
       geo = self.geo_params[idx]
       sid = int(self.simu_ids[idx]) # returns sid for labeling/debugging

       # IMPORTANT:
       # Dataset returns RAW geometry only.
       # Stage-II should compute TRAIN-only scaler after split to avoid leakage.
       geo_norm = geo

       return x, geo_norm, sid


# Helper: extract simulation IDs from a Subset produced by random_split (NEW)
def _subset_sids(dataset: PDNDataset, subset: Subset) -> np.ndarray:
   idxs = np.asarray(subset.indices, dtype=np.int64)
   sids = dataset.simu_ids[idxs].cpu().numpy().astype(int)
   return sids


# The DataLoader wraps a Dataset and handles batching, splitting, shuffling and parallel loading for you.
def get_dataloaders(
   data_dir: str,
   csv_file: str,
   batch_size: int = 64,
   subset_size: Optional[int] = None,
   seed: int = 42,
   repr_name: str = "ln_phase",  # pass representation tag for cache naming

   # (NEW) subset selection
   subset_sampling: str = "first",
   subset_seed: int = 42,

   # (NEW) geometry normalization
   # kept for compatibility with older notebook signature; dataset returns RAW geo
   geo_norm: str = "none",
   geo_scaler_path: Optional[str] = None,

   num_workers: int = 0,
   pin_memory: bool = False,
   return_split_info: bool = True,
):
   dataset = PDNDataset(
       data_dir=data_dir,
       csv_file=csv_file,
       subset_size=subset_size,
       repr_name=repr_name,
       subset_sampling=subset_sampling,
       subset_seed=subset_seed,
       geo_norm=geo_norm,
       geo_scaler_path=geo_scaler_path,
   )
   # Creates dataset
   n = len(dataset)

   # ---------------------------------------------------------
   # Ratio-based split (70/20/10) -> independent of subset size
   # ---------------------------------------------------------
   n_train = int(0.7 * n)
   n_val = int(0.2 * n)
   n_test = n - n_train - n_val

   # Deterministic split
   generator = torch.Generator().manual_seed(seed)
   train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)

   train_loader = DataLoader(
       train_ds,
       batch_size=batch_size,
       shuffle=True,              # train shuffled → better SGD training
       num_workers=num_workers,
       pin_memory=pin_memory,
   )
   val_loader = DataLoader(
       val_ds,
       batch_size=batch_size,
       shuffle=False,
       num_workers=num_workers,
       pin_memory=pin_memory,
   )
   test_loader = DataLoader(
       test_ds,
       batch_size=batch_size,
       shuffle=False,
       num_workers=num_workers,
       pin_memory=pin_memory,
   )

   if not return_split_info:
       return train_loader, val_loader, test_loader

   # Split info for notebook: print ranges + choose specific test IDs
   train_sids = _subset_sids(dataset, train_ds)
   val_sids = _subset_sids(dataset, val_ds)
   test_sids = _subset_sids(dataset, test_ds)

   split_info: Dict[str, Any] = {
       "n_total": int(n),
       "n_train": int(n_train),
       "n_val": int(n_val),
       "n_test": int(n_test),
       "train_sids": train_sids,
       "val_sids": val_sids,
       "test_sids": test_sids,
       "cache_path": dataset.cache_path,
       "cache_meta": getattr(dataset, "cache_meta", {}),
       "repr_name": repr_name,
       "seed": seed,

       # subset selection info
       "subset_sampling": subset_sampling,
       "subset_seed": int(subset_seed),
       "subset_ids_first50": dataset.selected_sids[:50].tolist(),
       "subset_ids_count": int(len(dataset.selected_sids)),

       # geometry normalization info (raw)
       "geo_norm": "raw_in_dataset",
       "geo_scaler_path": None,

       # NEW: freq grid info (useful for plotting + banded metrics)
       "n_freq": int(dataset.freq_hz.numel()) if getattr(dataset, "freq_hz", None) is not None else None,
       "f_start_hz": float(dataset.freq_hz[0].item()) if getattr(dataset, "freq_hz", None) is not None else None,
       "f_stop_hz": float(dataset.freq_hz[-1].item()) if getattr(dataset, "freq_hz", None) is not None else None,
   }

   return train_loader, val_loader, test_loader, split_info
