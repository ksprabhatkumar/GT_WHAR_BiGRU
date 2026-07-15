import os
import numpy as np
import torch
from tqdm import tqdm

def process_dsads():
    base_path = os.path.join("har_data", "DSADS", "data")
    all_data, all_labels, all_persons = [], [], []

    for activity_id in tqdm(range(1, 20), desc="Processing Activities"):
        for person_id in range(1, 9):
            for segment_id in range(1, 61):
                file_path = os.path.join(base_path, f"a{activity_id:02d}", f"p{person_id}", f"s{segment_id:02d}.txt")
                try:
                    data_matrix = np.loadtxt(file_path, delimiter=',')
                    data_reshaped = data_matrix.reshape(125, 5, 9)
                    all_data.append(data_reshaped)
                    all_labels.append(activity_id - 1)
                    all_persons.append(person_id) # Save the subject ID for LOSO
                except Exception:
                    pass

    X_tensor = torch.tensor(np.array(all_data), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(all_labels), dtype=torch.long)
    P_tensor = torch.tensor(np.array(all_persons), dtype=torch.long)

    os.makedirs("dataset/processed", exist_ok=True)
    torch.save(X_tensor, "dataset/processed/dsads_x.pt")
    torch.save(Y_tensor, "dataset/processed/dsads_y.pt")
    torch.save(P_tensor, "dataset/processed/dsads_p.pt")
    print(f"Saved: X {X_tensor.shape}, Y {Y_tensor.shape}, P {P_tensor.shape}")

if __name__ == "__main__":
    process_dsads()