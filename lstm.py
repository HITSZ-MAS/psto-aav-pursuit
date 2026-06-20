import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TrajectoryPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(TrajectoryPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        actual_input_dim = 3
        if input_dim > 3:
            actual_input_dim = input_dim
        self.lstm = nn.LSTM(actual_input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, historical_data_abs, mask=None):
        historical_pos_abs = historical_data_abs[..., :3]
        if mask is not None and mask.any():
            batch_size = mask.shape[0]
            first_valid_indices = torch.argmax(mask.to(torch.int), dim=1)
            start_pos = historical_pos_abs[
                torch.arange(batch_size), first_valid_indices
            ].unsqueeze(1)
        else:
            start_pos = historical_pos_abs[:, 0:1, :]
        historical_pos_rel = historical_pos_abs - start_pos
        if historical_data_abs.shape[-1] > 3:
            extra_features = historical_data_abs[..., 3:]
            lstm_input = torch.cat([historical_pos_rel, extra_features], dim=-1)
        else:
            lstm_input = historical_pos_rel
        if mask is not None and mask.any():
            lengths = mask.sum(dim=1).cpu()
            lengths = torch.clamp(lengths, min=1)
            packed_input = pack_padded_sequence(
                lstm_input, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed_input)
            lstm_out, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=historical_data_abs.shape[1],
            )
            batch_size = lstm_out.shape[0]
            last_seq_idxs = lengths - 1
            last_timestep_output = lstm_out[torch.arange(batch_size), last_seq_idxs]
        else:
            lstm_out, _ = self.lstm(lstm_input)
            last_timestep_output = lstm_out[:, -1, :]
        predicted_trajectory_rel = self.fc(last_timestep_output)
        num_future_steps = predicted_trajectory_rel.shape[-1] // 3
        predicted_trajectory_rel = predicted_trajectory_rel.view(
            predicted_trajectory_rel.shape[0], num_future_steps, 3
        )
        last_known_pos = historical_pos_abs[:, -1:, :]
        predicted_trajectory_abs = predicted_trajectory_rel + last_known_pos
        return predicted_trajectory_abs.flatten(start_dim=1)
