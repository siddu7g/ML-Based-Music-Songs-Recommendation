import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

DATA_PATH = r"C:\Users\sidat\OneDrive\Documents\SpotifyFeatures.csv"  # <-- CHANGE THIS

df = pd.read_csv(DATA_PATH)

print("Loaded rows:", len(df))
print(df.head())

# Create binary popularity target
df["popular"] = (df["popularity"] >= 50).astype(np.float32)

# Encode genre labels for genre head
genre_encoder = LabelEncoder()
df["genre_label"] = genre_encoder.fit_transform(df["genre"])

# FEATURES & PREPROCESSING
num_features = [
    "acousticness", "danceability", "duration_ms", "energy",
    "instrumentalness", "liveness", "loudness",
    "speechiness", "tempo", "valence"  
]

cat_features = ["genre", "key", "mode", "time_signature"]  

target_pop = "popular"
target_genre = "genre_label"
target_valence = "valence"

# ColumnTransformer: scale numeric, one-hot encode categorical
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

X = df[num_features + cat_features]
y_pop = df[target_pop].astype(np.float32).values
y_genre = df[target_genre].astype(np.int64).values
y_valence = df[target_valence].astype(np.float32).values

print("Applying preprocessing...")
X_processed = preprocess.fit_transform(X).toarray().astype(np.float32)  # dense for PyTorch

# TRAIN / VAL / TEST SPLIT
X_train, X_test, pop_train, pop_test, genre_train, genre_test, val_train, val_test = train_test_split(
    X_processed, y_pop, y_genre, y_valence, test_size=0.20, random_state=42
)

X_train, X_val, pop_train, pop_val, genre_train, genre_val, val_train, val_val = train_test_split(
    X_train, pop_train, genre_train, val_train, test_size=0.20, random_state=42
)

# Convert to torch tensors
X_train_t = torch.tensor(X_train)
pop_train_t = torch.tensor(pop_train).unsqueeze(1)
genre_train_t = torch.tensor(genre_train)
val_train_t = torch.tensor(val_train).unsqueeze(1)

X_val_t = torch.tensor(X_val)
pop_val_t = torch.tensor(pop_val).unsqueeze(1)
genre_val_t = torch.tensor(genre_val)
val_val_t = torch.tensor(val_val).unsqueeze(1)

X_test_t = torch.tensor(X_test)
pop_test_t = torch.tensor(pop_test).unsqueeze(1)
genre_test_t = torch.tensor(genre_test)
val_test_t = torch.tensor(val_test).unsqueeze(1)

train_ds = TensorDataset(X_train_t, pop_train_t, genre_train_t, val_train_t)
val_ds = TensorDataset(X_val_t, pop_val_t, genre_val_t, val_val_t)

train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=256)

# MULTI-TASK FNN MODEL
class MultiTaskNN(nn.Module):
    def __init__(self, input_dim, num_genres):
        super().__init__()
        # Shared body
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Popularity head (binary)
        self.pop_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # Genre head (multi-class)
        self.genre_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_genres)  # logits -> CrossEntropyLoss
        )
        # Valence head (regression)
        self.valence_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        shared_out = self.shared(x)
        pop_pred = self.pop_head(shared_out)
        genre_logits = self.genre_head(shared_out)
        valence_pred = self.valence_head(shared_out)
        return pop_pred, genre_logits, valence_pred

input_dim = X_processed.shape[1]
num_genres = len(df["genre_label"].unique())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = MultiTaskNN(input_dim, num_genres).to(device)

loss_pop = nn.BCELoss()
loss_genre = nn.CrossEntropyLoss()
loss_valence = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# TRAINING LOOP
epochs = 15
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for xb, y_pop_b, y_genre_b, y_val_b in train_dl:
        xb = xb.to(device)
        y_pop_b = y_pop_b.to(device)
        y_genre_b = y_genre_b.to(device)
        y_val_b = y_val_b.to(device)

        optimizer.zero_grad()
        pred_pop, pred_genre, pred_val = model(xb)

        l1 = loss_pop(pred_pop, y_pop_b)
        l2 = loss_genre(pred_genre, y_genre_b)
        l3 = loss_valence(pred_val, y_val_b)

        loss = l1 + l2 + l3
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_dl)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        v_pop, v_genre, v_val = model(X_val_t.to(device))
        val_loss = (
            loss_pop(v_pop, pop_val_t.to(device)) +
            loss_genre(v_genre, genre_val_t.to(device)) +
            loss_valence(v_val, val_val_t.to(device))
        ).item()
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

# TEST EVALUATION
model.eval()
with torch.no_grad():
    t_pop, t_genre, t_val = model(X_test_t.to(device))

pop_pred_class = (t_pop.cpu() > 0.5).float()
genre_pred_class = torch.argmax(t_genre.cpu(), dim=1)
val_mse = ((t_val.cpu() - val_test_t)**2).mean().item()

pop_acc = accuracy_score(pop_test_t, pop_pred_class)
genre_acc = accuracy_score(genre_test_t, genre_pred_class)

print("\n===== MULTI-TASK TEST RESULTS =====")
print("Popularity Accuracy:", pop_acc)
print("Genre Accuracy:", genre_acc)
print("Valence MSE:", val_mse)

#  PLOT TRAINING CURVES (optional)
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Multi-Task Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# INTERACTIVE RECOMMENDER

def interactive_recommender():
    print("\n=== 🎵 MUSIC RECOMMENDER SYSTEM (FNN Powered) 🎵 ===")
    print("Answer the following questions to get personalized song suggestions.\n")

    try:
        danceability = float(input("How danceable should the song be? (0 to 1): "))
        energy       = float(input("Preferred energy level? (0 to 1): "))
        valence_pref = float(input("Preferred emotional positivity (valence)? (0 to 1): "))

        # Show some sample genres
        unique_genres = df["genre"].unique()
        print("\nAvailable Genres (first 20 shown):")
        print(unique_genres[:20])
        genre_name   = input("Enter your preferred genre (exactly as shown above if possible): ")

        top_k        = int(input("How many recommendations do you want? (e.g., 10): "))

    except Exception as e:
        print("Invalid input, please try again.", e)
        return

    # Build a synthetic user preference row
    user_dict = {
        "acousticness": 0.2,
        "danceability": danceability,
        "duration_ms": 200000,
        "energy": energy,
        "instrumentalness": 0.0,
        "liveness": 0.15,
        "loudness": -5,
        "speechiness": 0.05,
        "tempo": 120,
        "valence": valence_pref,
        "genre": genre_name,
        "key": "C",
        "mode": "Major",
        "time_signature": "4/4"
    }

    user_df = pd.DataFrame([user_dict])

    # Preprocess user vector
    user_processed = preprocess.transform(user_df).toarray().astype(np.float32)
    user_tensor = torch.tensor(user_processed).to(device)

    model.eval()
    with torch.no_grad():
        user_pop_pred, _, _ = model(user_tensor)

    user_score = float(user_pop_pred.cpu().item())
    print(f"\nEstimated likability score for your preferences: {user_score:.3f}")

    # Score all songs in the dataset (reuse X_processed)
    X_all_t = torch.tensor(X_processed).to(device)
    with torch.no_grad():
        all_pop_scores, _, _ = model(X_all_t)

    scores = all_pop_scores.cpu().numpy().flatten()
    df_rec = df.copy()
    df_rec["likability"] = scores

    # Filter > 0.8 likelihood
    recommended = df_rec[df_rec["likability"] >= 0.80].sort_values(by="likability", ascending=False)

    print(f"\n=== 🎧 Top {top_k} Recommended Songs For You ===\n")
    if "track_name" in df_rec.columns and "artist_name" in df_rec.columns:
        cols_to_show = ["track_name", "artist_name", "genre", "likability"]
    else:
        cols_to_show = ["genre", "likability"]
    print(recommended[cols_to_show].head(top_k))
    print("\nDone!\n")

if __name__ == "__main__":
    interactive_recommender()
