import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Encoder, Decoder


class Game(nn.Module):
    def __init__(self, encoder : Encoder, decoder : Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.register_buffer(
            "logV",
            torch.log(torch.tensor(self.encoder.vocab_size, dtype=torch.float32)),
            persistent=True,
        )

        self.loss_fn = lambda a, b: F.mse_loss(a, b, reduction='none').sum(dim=1)

    def forward(self, x, tau, beta=None, free_bits=None):
        with torch.autocast('cuda', torch.bfloat16):
            logits, message = self.encoder(x, tau)
            receiver_output = self.decoder(message)
            
        with torch.autocast('cuda', torch.float32):
            recon_loss = self.loss_fn(x.float(), receiver_output[:, -1].float()).mean()
            
            q_logprobs = logits.log_softmax(dim=-1)
            q_probs = q_logprobs.exp()

            H_Z_given_X = -(q_probs * q_logprobs).sum(dim=-1)
            KL_step = self.logV - H_Z_given_X  # [B,T]
            if free_bits is not None:
                KL_step_fb = F.relu(KL_step - free_bits)
            else:
                KL_step_fb = KL_step
            KL_vocab_term = KL_step_fb.sum(dim=-1).mean()

            if beta is not None:
                loss = recon_loss + beta * KL_vocab_term
            else:
                loss = recon_loss

            with torch.no_grad():
                H_Z_given_X_total = H_Z_given_X.sum(dim=-1).mean()
                q_marg = q_probs.mean(dim=0)
                H_Z = -(q_marg * (q_marg + 1e-12).log()).sum()
                I_XZ = H_Z - H_Z_given_X_total
                KL_q_p = (self.logV + (q_marg * ((q_marg + 1e-12).log())).sum(dim=1)).sum()
                gs_entropy = -(message * (message + 1e-12).log()).sum(dim=-1).sum(dim=1).mean()
                metrics = {
                    'loss': loss,
                    'recon_loss': recon_loss.detach(),
                    'KL_term': KL_vocab_term.detach(),
                    'H_Z_given_X': H_Z_given_X_total,
                    'H_Z': H_Z ,
                    'I_XZ': I_XZ,
                    'KL_q_p': KL_q_p,
                    'gs_entropy': gs_entropy
                }
                if free_bits is not None:
                    metrics['frac_under_fb'] = (KL_step < free_bits).float().mean(dim=0).sum()
                if beta is not None:
                    metrics['vocab_loss'] = beta * KL_vocab_term.detach()
                
        return loss, metrics
    

class VarlenGame(nn.Module):
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            metadata_mode=None,
            num_artist_classes=None,
            num_album_classes=None,
            artist_loss_weight=0.0,
            album_loss_weight=0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.metadata_mode = metadata_mode
        self.artist_loss_weight = float(artist_loss_weight)
        self.album_loss_weight = float(album_loss_weight)

        if metadata_mode not in (None, "aux", "prefix"):
            raise ValueError("metadata_mode must be one of: None, 'aux', 'prefix'")
        if metadata_mode is not None:
            if num_artist_classes is None or int(num_artist_classes) <= 0:
                raise ValueError("num_artist_classes must be positive when metadata supervision is enabled")
            if num_album_classes is None or int(num_album_classes) <= 0:
                raise ValueError("num_album_classes must be positive when metadata supervision is enabled")
            self.artist_head = nn.Linear(encoder.hidden_size, int(num_artist_classes))
            self.album_head = nn.Linear(encoder.hidden_size, int(num_album_classes))

        self.register_buffer(
            "logV",
            torch.log(torch.tensor(self.encoder.vocab_size, dtype=torch.float32)),
            persistent=True,
        )

        self.loss_fn = lambda a, b: F.mse_loss(a, b, reduction='none').sum(dim=1)

    def _metadata_loss(self, message, smoothed_alive, artist_labels, album_labels):
        if self.metadata_mode is None:
            return None, {}
        if artist_labels is None or album_labels is None:
            raise ValueError("artist_labels and album_labels are required for metadata supervision")

        codebook_embeddings = torch.stack([
            message[:, step].float() @ self.encoder._get_codebook(step).float()
            for step in range(self.encoder.maxlen)
        ], dim=1)

        if self.metadata_mode == "aux":
            metadata_repr = (smoothed_alive[..., None] * codebook_embeddings).sum(dim=1)
            artist_repr = metadata_repr
            album_repr = metadata_repr
        else:
            artist_repr = codebook_embeddings[:, 0]
            album_repr = codebook_embeddings[:, :min(2, self.encoder.maxlen)].sum(dim=1)

        artist_logits = self.artist_head(artist_repr)
        album_logits = self.album_head(album_repr)
        artist_loss = F.cross_entropy(artist_logits, artist_labels)
        album_loss = F.cross_entropy(album_logits, album_labels)
        metadata_loss = self.artist_loss_weight * artist_loss + self.album_loss_weight * album_loss

        with torch.no_grad():
            metrics = {
                "metadata_loss": metadata_loss.detach(),
                "artist_loss": artist_loss.detach(),
                "album_loss": album_loss.detach(),
                "artist_accuracy": (artist_logits.argmax(dim=-1) == artist_labels).float().mean(),
                "album_accuracy": (album_logits.argmax(dim=-1) == album_labels).float().mean(),
            }
        return metadata_loss, metrics

    def forward(
            self,
            x,
            tau,
            length_cost,
            beta=None,
            free_bits=None,
            artist_labels=None,
            album_labels=None,
    ):
        with torch.autocast('cuda', torch.bfloat16):
            logits, message, length_logits, survival_logits = self.encoder(x, tau)
            receiver_output = self.decoder(message)
            
        with torch.autocast('cuda', torch.float32):
            T = self.encoder.maxlen

            length_probs = length_logits.softmax(dim=-1)
            length_logprobs = length_logits.log_softmax(dim=-1)
            smoothed_length_probs = 0.9 * length_probs + 0.1 * (1.0 / T)
            smoothed_alive = length_probs.flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))

            alive = survival_logits.exp().clamp(min=0.0, max=1.0)
            alive_mean = alive.mean(dim=0)

            recon_loss = 0.0
            for step in range(T):
                step_loss = self.loss_fn(x.float(), receiver_output[:, step, ...].float())
                recon_loss = recon_loss + (smoothed_length_probs[:, step] * step_loss).mean()

            q_logprobs = logits.log_softmax(dim=-1)
            q_probs = q_logprobs.exp()

            H_Z_given_X = -(q_probs * q_logprobs).sum(dim=-1)
            KL_step = self.logV - H_Z_given_X  # [B,T]
            if free_bits is not None:
                KL_step_fb = F.relu(KL_step - free_bits)
            else:
                KL_step_fb = KL_step
            KL_weighted_step_fb = smoothed_alive * KL_step_fb
            KL_vocab_term = KL_weighted_step_fb.sum(dim=-1).mean()

            E_L = (length_probs * torch.arange(1, T + 1, device=length_probs.device, dtype=length_probs.dtype)[None, :]).sum(dim=-1).mean()
            length_entropy = -(length_probs * length_logprobs).sum(dim=-1).mean()
            KL_length_term = -length_entropy + length_cost * E_L

            if beta is not None:
                loss = recon_loss + beta * (KL_vocab_term + KL_length_term)
            else:
                loss = recon_loss
            metadata_loss, metadata_metrics = self._metadata_loss(
                message,
                smoothed_alive,
                artist_labels,
                album_labels,
            )
            if metadata_loss is not None:
                loss = loss + metadata_loss

            with torch.no_grad():
                H_Z_given_X_total = (alive * H_Z_given_X).sum(dim=-1).mean()

                q_marg = q_probs.mean(dim=0)  # [V]
                H_Z_step = -(q_marg * (q_marg + 1e-12).log()).sum(dim=-1)
                H_Z = (alive_mean * H_Z_step).sum()
    
                I_XZ = H_Z - H_Z_given_X_total

                KL_q_p = (alive_mean * (self.logV + (q_marg * ((q_marg + 1e-12).log())).sum(dim=1))).sum()

                gs_entropy = (alive * -(message * (message + 1e-12).log()).sum(dim=-1)).sum(dim=1).mean()

                metrics = {
                    'loss': loss,
                    'recon_loss': recon_loss.detach(),
                    'KL_term': KL_vocab_term.detach(),
                    'H_Z_given_X': H_Z_given_X_total,
                    'H_Z': H_Z ,
                    'I_XZ': I_XZ,
                    'KL_q_p': KL_q_p,
                    'gs_entropy': gs_entropy,
                    'E_L': E_L.detach(),
                    'length_entropy': length_entropy
                }
                if free_bits is not None:
                    metrics['frac_under_fb'] = (KL_step < free_bits).float().mean(dim=0).sum()
                if beta is not None:
                    metrics['vocab_loss'] = beta * KL_vocab_term.detach()
                    metrics['length_loss'] = beta * KL_length_term.detach()
                metrics.update(metadata_metrics)
                
        return loss, metrics
