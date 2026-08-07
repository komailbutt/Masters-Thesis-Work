# Sources/models.py

import torch
import torch.nn as nn


class CNNAutoencoder(nn.Module):
  def __init__(self, latent_dim=16):
      super().__init__()

      # ---------------- ENCODER ----------------
      # Input size: [Batch, 2, 334]
      # Goal: compress frequency axis while increasing channels (feature richness)
      self.encoder = nn.Sequential(
          # Layer 1: 334 -> 167
          # Conv1d(in_channels=2, out_channels=16, kernel=3, stride=2, padding=1)
          nn.Conv1d(2, 16, 3, stride=2, padding=1),
          nn.ReLU(),

          # Layer 2: 167 -> 84
          nn.Conv1d(16, 32, 3, stride=2, padding=1),
          nn.ReLU(),

          # Layer 3: 84 -> 42
          nn.Conv1d(32, 64, 3, stride=2, padding=1),
          nn.ReLU(),

          # Flatten [Batch, 64, 42] -> [Batch, 64*42]
          nn.Flatten(),
      )

      # ---------------- LATENT ----------------
      # After encoder: 64 filters * 42 spatial points = 2688 features
      self.fc_latent = nn.Linear(64 * 42, latent_dim)

      # Decoder "seed": map latent back to feature grid [Batch, 64, 42]
      self.fc_dec = nn.Linear(latent_dim, 64 * 42)

      # ---------------- DECODER ----------------
      # Reverse the downsampling using ConvTranspose1d
      self.decoder = nn.Sequential(
          # Reverse Layer 3: 42 -> 84 (needs output_padding=1)
          nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=1),
          nn.ReLU(),

          # Reverse Layer 2: 84 -> 167
          # NOTE: output_padding is not required here for correct length
          nn.ConvTranspose1d(32, 16, 3, stride=2, padding=1),
          nn.ReLU(),

          # Reverse Layer 1: 167 -> 334 (needs output_padding=1)
          nn.ConvTranspose1d(16, 2, 3, stride=2, padding=1, output_padding=1),
      )

  # --------------------------------------------------
  # NEW: encode()  → returns latent code z
  # --------------------------------------------------
  def encode(self, x: torch.Tensor) -> torch.Tensor:
      """
      Encoder forward pass only.

      Args:
          x : [Batch, 2, 334] (typically normalized input for Stage-1)
      Returns:
          z : [Batch, latent_dim]
      """
      h = self.encoder(x)        # [B, 64*42]
      z = self.fc_latent(h)      # [B, latent_dim]
      return z

  # --------------------------------------------------
  # NEW: decode()  → reconstruct from latent z
  # --------------------------------------------------
  def decode(self, z: torch.Tensor) -> torch.Tensor:
      """
      Decoder forward pass only.

      Args:
          z : [Batch, latent_dim]
      Returns:
          x_recon : [Batch, 2, 334] (normalized domain)
      """
      x_feat = self.fc_dec(z).view(-1, 64, 42)  # [B, 64, 42]
      x_recon = self.decoder(x_feat)            # [B, 2, 334]
      return x_recon

  def forward(self, x: torch.Tensor):
      # Keep original behavior for Stage-1 training/eval compatibility:
      # returns (reconstruction, latent)
      z = self.encode(x)
      x_recon = self.decode(z)
      return x_recon, z


class LSTMAutoencoder(nn.Module):
  """
  LSTM Autoencoder for impedance profiles.

  Input:
      x : [Batch, C, 334]  (channels-first as used in your dataset)
          C = input_dim
          - ln_phase   -> C=2  (ln|Z|, phase)
          - ln_sincos  -> C=3  (ln|Z|, sin(phase), cos(phase))  (if use this representation)

  Internally for LSTM:
      we treat frequency points as a sequence:
          seq_len = 334
          input_dim = C
      so we transpose to:
          x_seq : [Batch, 334, C]

  Output:
      x_recon : [Batch, C, 334]  (same as input shape)
      z       : [Batch, latent_dim]
  """

  def __init__(
      self,
      latent_dim: int = 25,
      hidden_dim: int = 128,
      num_layers: int = 2,
      dropout: float = 0.1,
      bidirectional: bool = True,
      seq_len: int = 334,
      input_dim: int = 2,

      # ---------------- NEW (Step 1) ----------------
      # Positional input for decoder (time-aware decoder)
      # We will CONCAT [token , pos_emb(t)] at each time step
      decoder_pos_dim: int = 16,
  ):
      super().__init__()

      self.latent_dim = int(latent_dim)
      self.hidden_dim = int(hidden_dim)
      self.num_layers = int(num_layers)
      self.dropout = float(dropout)
      self.bidirectional = bool(bidirectional)

      self.seq_len = int(seq_len)
      self.input_dim = int(input_dim)

      # NEW: decoder positional embedding size
      self.decoder_pos_dim = int(decoder_pos_dim)

      # ---------------- ENCODER (LSTM) ----------------
      # Reads sequence x_seq: [B, T, C] and compresses it to a hidden state
      #
      # Note:
      # - batch_first=True means input/output are [B, T, D]
      # - dropout in nn.LSTM is applied between layers (only if num_layers > 1)
      self.enc_lstm = nn.LSTM(
          input_size=self.input_dim,
          hidden_size=self.hidden_dim,
          num_layers=self.num_layers,
          batch_first=True,
          dropout=self.dropout if self.num_layers > 1 else 0.0,
          bidirectional=self.bidirectional,
      )

      enc_out_dim = self.hidden_dim * (2 if self.bidirectional else 1)

      # ---------------- LATENT ----------------
      # Map last hidden state -> latent vector z
      self.fc_latent = nn.Linear(enc_out_dim, self.latent_dim)

      # Map latent z back -> decoder token (base conditioning)
      self.fc_dec_in = nn.Linear(self.latent_dim, enc_out_dim)

      # ---------------- NEW (Step 1): POSITION ----------------
      # Learnable positional embeddings for each time step: [T, pos_dim]
      # This makes the decoder time-aware (otherwise repeating the same token is ambiguous).
      self.pos_emb = nn.Embedding(self.seq_len, self.decoder_pos_dim)

      # Decoder will receive at each time step:
      #   dec_in(t) = concat( token , pos_emb(t) )
      dec_input_dim = enc_out_dim + self.decoder_pos_dim

      # ---------------- DECODER (LSTM) ----------------
      self.dec_lstm = nn.LSTM(
          input_size=dec_input_dim,          # <-- CHANGED (now token+position)
          hidden_size=self.hidden_dim,
          num_layers=self.num_layers,
          batch_first=True,
          dropout=self.dropout if self.num_layers > 1 else 0.0,
          bidirectional=self.bidirectional,
      )

      dec_out_dim = self.hidden_dim * (2 if self.bidirectional else 1)

      # Final projection per time step: hidden -> C channels
      self.fc_out = nn.Linear(dec_out_dim, self.input_dim)

  # --------------------------------------------------
  # NEW: encode()  → returns latent code z
  # --------------------------------------------------
  def encode(self, x: torch.Tensor) -> torch.Tensor:
      """
      Encoder forward pass only.

      Args:
          x : [Batch, C, 334] (typically normalized input for Stage-1)
      Returns:
          z : [Batch, latent_dim]
      """
      if x.ndim != 3:
          raise ValueError(f"Expected x to be 3D [B, C, {self.seq_len}], got shape {tuple(x.shape)}")
      if x.shape[1] != self.input_dim:
          raise ValueError(
              f"Expected channel dim={self.input_dim}, got x.shape[1]={x.shape[1]}. "
              f"(Check RUN_CFG['repr'] and model input_dim.)"
          )
      if x.shape[2] != self.seq_len:
          raise ValueError(f"Expected seq_len={self.seq_len}, got x.shape[2]={x.shape[2]}")

      # [B, C, T] -> [B, T, C]
      x_seq = x.transpose(1, 2).contiguous()

      # LSTM outputs:
      #   out: [B, T, H*dir]
      #   (h_n, c_n): h_n is [num_layers*dir, B, H]
      out, (h_n, c_n) = self.enc_lstm(x_seq)

      # Take last layer hidden state (and concat directions if bidirectional)
      if self.bidirectional:
          # last layer has two directions at indices [-2] and [-1]
          h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, 2H]
      else:
          h_last = h_n[-1]  # [B, H]

      z = self.fc_latent(h_last)  # [B, latent_dim]
      return z

  # --------------------------------------------------
  # NEW: decode()  → reconstruct from latent z
  # --------------------------------------------------
  def decode(self, z: torch.Tensor) -> torch.Tensor:
      """
      Decoder forward pass only.

      Args:
          z : [Batch, latent_dim]
      Returns:
          x_recon : [Batch, C, 334] (normalized domain)
      """
      if z.ndim != 2 or z.shape[1] != self.latent_dim:
          raise ValueError(f"Expected z shape [B, {self.latent_dim}], got {tuple(z.shape)}")

      B = z.shape[0]
      device = z.device

      # Base token from latent
      token = self.fc_dec_in(z)  # [B, enc_out_dim]

      # Repeat token across all time steps: [B, T, enc_out_dim]
      token_seq = token.unsqueeze(1).repeat(1, self.seq_len, 1)

      # NEW: positional embeddings: [T] -> [T, pos_dim] -> [B, T, pos_dim]
      pos_ids = torch.arange(self.seq_len, device=device)                 # [T]
      pos_seq = self.pos_emb(pos_ids).unsqueeze(0).repeat(B, 1, 1)        # [B, T, pos_dim]

      # NEW (Step 1): decoder input = concat(token, position)
      dec_in = torch.cat([token_seq, pos_seq], dim=-1)                    # [B, T, enc_out_dim+pos_dim]

      dec_out, _ = self.dec_lstm(dec_in)                                  # [B, T, H*dir]
      x_seq_recon = self.fc_out(dec_out)                                  # [B, T, C]

      # [B, T, C] -> [B, C, T]
      x_recon = x_seq_recon.transpose(1, 2).contiguous()
      return x_recon

  def forward(self, x: torch.Tensor):
      # Keep original behavior for Stage-1 training/eval compatibility:
      # returns (reconstruction, latent)
      z = self.encode(x)
      x_recon = self.decode(z)
      return x_recon, z