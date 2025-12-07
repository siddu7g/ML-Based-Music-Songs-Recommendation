1. The model is trained on SpotifyFeatures dataset from Kaggle

2. The network follows the following architecture: popularity, danceability & Valence as blocks.
![System Integration](FcNN/Screenshot.png)

3. User Centric Design; Likability Score Prediction (Output): Represents a learned probability based on audio and metadata features

🎵 MUSIC RECOMMENDER SYSTEM (FNN Powered) 🎵 
Answer the following questions to get personalized song suggestions.

How danceable should the song be? (0 to 1): 0.5
Preferred energy level? (0 to 1): 0.7
Preferred emotional positivity (valence)? (0 to 1): 0.8

Available Genres (first 20 shown):
['Movie' 'R&B' 'A Capella' 'Alternative' 'Country' 'Dance' 'Electronic'
 'Anime' 'Folk' 'Blues' 'Opera' 'Hip-Hop' "Children's Music"
 'Children’s Music' 'Rap' 'Indie' 'Classical' 'Pop' 'Reggae' 'Reggaeton']
Enter your preferred genre (exactly as shown above if possible): Blues
How many recommendations do you want? (e.g., 10): 7

Estimated likability score for your preferences: 0.074

=== 🎧 Top 7 Recommended Songs For You ===
```bash

                    track_name          artist_name genre  likability
150858  Catarino y los Rurales          El Fantasma   Pop    0.999867
109767                   Mi 45          El Fantasma   Pop    0.999861
112230   No Te Ilusiones Tanto        Adriel Favela   Pop    0.999852
112443               La Piedra          El Fantasma   Pop    0.999842
111059                 Imma Be  The Black Eyed Peas   Pop    0.999827
113742   El Día De Los Muertos       Alfredo Olivas   Pop    0.999810
150415        Me Está Gustando  Banda Los Recoditos   Pop    0.999808
```
