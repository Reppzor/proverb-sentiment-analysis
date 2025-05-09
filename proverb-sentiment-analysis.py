import pandas as pd
from sklearn.metrics import cohen_kappa_score
from itertools import combinations

# Κάθε αρχείο περιέχει δεδομένα που έχουν καταχωρηθεί από διαφορετικούς annotators.
file1_path = "/content/1296_f3662401.xlsx"
file2_path = "/content/1296_f3662402.xlsx"
file3_path = "/content/1296_f3662403.xlsx"
file4_path = "/content/1296_f3662404.xlsx"
file5_path = "/content/1296_f3662405.xlsx"
file7_path = "/content/1296_f3662407.xlsx"
file10_path = "/content/1296_f3662410.xlsx"
file11_path = "/content/1296_f3662411.xlsx"
file12_path = "/content/1296_f3662412.xlsx"
file14_path = "/content/1296_f3662414.xlsx"
file16_path = "/content/1296_f3662416.xlsx"
file17_path = "/content/1296_f3662417.xlsx"


df1 = pd.read_excel(file1_path)
df2 = pd.read_excel(file2_path)
df3 = pd.read_excel(file3_path)
df4 = pd.read_excel(file4_path)
df5 = pd.read_excel(file5_path)
df7 = pd.read_excel(file7_path)
df10 = pd.read_excel(file10_path)
df11 = pd.read_excel(file11_path)
df12 = pd.read_excel(file12_path)
df14 = pd.read_excel(file14_path)
df16 = pd.read_excel(file16_path)
df17 = pd.read_excel(file17_path)

# Σε κάθε DataFrame προσθέτουμε μία στήλη για τον αναλυτή που έχει προσθέσει τα δεδομένα,
# ώστε να ξέρουμε ποιος αναλυτής έχει προσδιορίσει ποιο συναίσθημα.
df1["Annotator"] = "Annotator_1"
df2["Annotator"] = "Annotator_2"
df3["Annotator"] = "Annotator_3"
df4["Annotator"] = "Annotator_4"
df5["Annotator"] = "Annotator_5"
df7["Annotator"] = "Annotator_6"
df10["Annotator"] = "Annotator_7"
df11["Annotator"] = "Annotator_8"
df12["Annotator"] = "Annotator_9"
df14["Annotator"] = "Annotator_10"
df16["Annotator"] = "Annotator_11"
df17["Annotator"] = "Annotator_12"

# Εδώ συγχωνεύουμε όλα τα DataFrames που περιέχουν δεδομένα από διαφορετικούς αναλυτές σε ένα ενιαίο DataFrame.
merged_df = pd.concat([df1, df2, df3, df4, df5, df7, df10, df11, df12, df14, df16, df17], ignore_index=True)
print("Merged DataFrame Sample:")
print(merged_df.head())

# Από το συγχωνευμένο DataFrame δημιουργούμε νέες DataFrames για
# κάθε χαρακτηριστικό (Sentiment, Emotion, Memeability), ώστε να μπορέσουμε να επεξεργαστούμε και να αναλύσουμε τα δεδομένα τους ξεχωριστά.
sentiment_df = merged_df[['Proverb', 'Annotator', 'Sentiment']].copy()
print("\nSentiment DataFrame Sample:")
print(sentiment_df.head())

emotion_df = merged_df[['Proverb', 'Annotator', 'Emotion']].copy()
print("\nEmotion DataFrame Sample:")
print(emotion_df.head())

memeability_df = merged_df[['Proverb', 'Annotator', 'Memeability']].copy()
print("\nMemeability DataFrame Sample:")
print(memeability_df.head())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Εδώ εξάγουμε όλες τις μοναδικές παροιμίες από τη στήλη "Proverb", αγνοώντας τις κενές τιμές.
proverbs = merged_df['Proverb'].dropna().unique()

# Χρησιμοποιούμε το TfidfVectorizer για να μετατρέψουμε τις παροιμίες σε διανύσματα, βασισμένα στη συχνότητα των λέξεων.
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(proverbs)

# Υπολογισμός της ομοιότητας του συνημίτονου (Cosine Similarity) για όλα τα ζεύγη των παροιμιών
cosine_sim = cosine_similarity(X)


# Ορίζουμε ένα όριο ομοιότητας κατόπιν δοκιμής και επαναξιολόγησης των δεδομένων για να εντοπίσουμε τις διπλότυπες παροιμίες.
threshold = 0.6

# Αρχικοποίηση ενός συνόλου για τις μοναδικές παροιμίες και μιας λίστας για τις τελικές παροιμίες
unique_proverbs = set()
final_proverbs = []

# Αρχικοποίηση μιας λίστας για τις παραλειπόμενες παροιμίες
skipped_proverbs = []


# Συγκρίνουμε κάθε παροιμία με τις μοναδικές και αν βρούμε ομοιότητα πάνω από το όριο, την παραλείπουμε.
for i in range(len(proverbs)):
    is_duplicate = False
    for unique_proverb in unique_proverbs:
        # Υπολογίζουμε την ομοιότητα του συνημίτονου μεταξύ της τρέχουσας παροιμίας και μιας μοναδικής
        similarity = cosine_similarity(vectorizer.transform([proverbs[i]]), vectorizer.transform([unique_proverb]))[0][0]
        if similarity > threshold:
            is_duplicate = True
            # Προσθέτουμε την διπλότυπη παροιμία στη λίστα skipped_proverbs
            skipped_proverbs.append((proverbs[i], unique_proverb, similarity))
            break
    if not is_duplicate:
        final_proverbs.append(proverbs[i])
        unique_proverbs.add(proverbs[i])

#Φιλτράρουμε το αρχικό DataFrame για να κρατήσουμε μόνο τις μη διπλότυπες παροιμίες
merged_df_no_duplicates = merged_df[merged_df['Proverb'].isin(final_proverbs)]

# Εμφάνιση του καθαρισμένου DataFrame (χωρίς διπλότυπες παροιμίες)
print("\nCleaned DataFrame (Duplicates Removed):")
print(merged_df_no_duplicates.head())

#Εκτύπωση των παραλειπόμενων παροιμιών (διπλότυπες)
print("\nList of Skipped Proverbs (Duplicates):")
for skipped in skipped_proverbs:
    print(f"Skipped Proverb: {skipped[0]} (similar to: {skipped[1]}) with similarity {skipped[2]:.2f}")

# Μετατροπή του Sentiment σε δυαδική μορφή
sentiment_df["Is_Neutral"] = sentiment_df["Sentiment"].apply(lambda x: 1 if x == "Neutral" else 0)
print("\nSentiment DataFrame with Is_Neutral Column Sample:")
print(sentiment_df.head())

# Χρησιμοποιούμε την μέθοδο pivot_table για να δημιουργήσουμε έναν πίνακα συμφωνίας όπου κάθε γραμμή αντιπροσωπεύει μια παροιμία
# και κάθε στήλη έναν αναλυτή, με την τιμή "Is_Neutral" που δείχνει αν η παροιμία θεωρήθηκε ουδέτερη.
agreement_data = sentiment_df.pivot_table(index="Proverb", columns="Annotator", values="Is_Neutral", aggfunc='first').dropna()
print("\nAgreement Data Sample:")
print(agreement_data.head())

# Υπολογισμός του ποσοστού συμφωνίας
num_agreements = (agreement_data.nunique(axis=1) == 1).sum()
total_cases = agreement_data.shape[0]
percentage_agreement = num_agreements / total_cases * 100

# Για κάθε ζεύγος αναλυτών, υπολογίζουμε το σκορ Cohen's Kappa για να δούμε πόσο συμφωνούν οι αναλυτές μεταξύ τους.
kappa_scores = {}
annotators = agreement_data.columns.tolist()
for annotator1, annotator2 in combinations(annotators, 2):
    kappa = cohen_kappa_score(agreement_data[annotator1], agreement_data[annotator2])
    kappa_scores[f"{annotator1} vs {annotator2}"] = kappa

# Εμφάνιση των αποτελεσμάτων συμφωνίας
print(f"\nPercentage Agreement: {percentage_agreement:.2f}%")
print("Cohen's Kappa Scores:")
for pair, score in kappa_scores.items():
    print(f"{pair}: {score:.2f}")

# Δημιουργία ενός DataFrame για τα σκορ Cohen's Kappa
kappa_matrix = pd.DataFrame(index=annotators, columns=annotators, dtype=float)


# Για κάθε ζεύγος αναλυτών, γεμίζουμε τον πίνακα με τα αντίστοιχα σκορ Cohen's Kappa.
for key, kappa in kappa_scores.items():
    annotator1, annotator2 = key.split(" vs ")  # Διαχωρίζουμε τα ονόματα των αναλυτών από το ζεύγος αναλυτών
    kappa_matrix.loc[annotator1, annotator2] = kappa
    kappa_matrix.loc[annotator2, annotator1] = kappa  # Ο πίνακας είναι συμμετρικός, οπότε η τιμή επαναλαμβάνεται και από την αντίθετη κατεύθυνση.
np.fill_diagonal(kappa_matrix.values, 1)  # Θέτουμε τη διαγώνιο στον πίνακα να έχει τιμή 1 (τέλεια συμφωνία με τον εαυτό του)

# Δημιουργούμε μια μάσκα για να αποκλείσουμε τις συγκρίσεις με τον ίδιο τον αναλυτή στη διαγώνιο του πίνακα.
mask = np.eye(len(annotators), dtype=bool)


# Παρουσιάζουμε heatmap για τη συμφωνία των αναλυτών, αλλά χωρίς να εμφανίζονται οι συγκρίσεις με τον εαυτό τους.
plt.figure(figsize=(10, 8))
sns.heatmap(kappa_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, mask=mask)
plt.title("Cohen's Kappa Heatmap: Neutral Sentiment Agreement (Excluding Self)")
plt.show()

# Βρίσκουμε το ζεύγος αναλυτών με τη μεγαλύτερη συμφωνία
max_pair = max(kappa_scores, key=kappa_scores.get)
max_score = kappa_scores[max_pair]
print(f"\nMost Agreeing Annotators: {max_pair} with Kappa = {max_score:.2f}")

# Ομαδοποιούμε τα δεδομένα βάσει του 'Proverb' και μετράμε πόσες φορές το 'Is_Neutral' είναι 1 (δηλαδή, ουδέτερο).
neutral_counts = sentiment_df.groupby("Proverb")["Is_Neutral"].sum().reset_index()

# Ταξινομούμε τα αποτελέσματα σε αύξουσα σειρά (όσα έχουν λιγότερες ουδέτερες ταξινομήσεις πρώτα)
# Αυτό θα μας επιτρέψει να δούμε τα προβλήματα που χαρακτηρίζονται λιγότερο ως ουδέτερα.
neutral_counts_sorted = neutral_counts.sort_values(by="Is_Neutral", ascending=True)

# Επιλέγουμε τα 10 πρώτα προβλήματα με τις λιγότερες ουδέτερες ταξινομήσεις
least_neutral_proverbs = neutral_counts_sorted.head(10)

# Εμφάνιση των αποτελεσμάτων
print("\nTop 10 Proverbs Least Characterized as Neutral:")
print(least_neutral_proverbs)

# Υπολογίζουμε το άθροισμα της στήλης 'Memeability'.
memeability_scores = merged_df.groupby("Proverb")["Memeability"].sum().reset_index()

# Ταξινομούμε τα αποτελέσματα σε φθίνουσα σειρά (όσα έχουν τις υψηλότερες βαθμολογίες memeability πρώτα)
memeability_sorted = memeability_scores.sort_values(by="Memeability", ascending=False)

# Επιλέγουμε τις 10 παροιμίες με τις υψηλότερες βαθμολογίες memeability
most_memeable_proverbs = memeability_sorted.head(10)

# Εμφάνιση των αποτελεσμάτων
print("\nTop 10 Most Memeable Proverbs:")
print(most_memeable_proverbs)

from itertools import combinations

# Ομαδοποιούμε τα δεδομένα με βάση την παροιμία και συλλέγουμε όλες τις ετικέτες Emotion σε λίστα
# Για κάθε παροιμία αποθηκεύουμε όλες τις ετικέτες Emotion που την αφορούν.
emotion_annotations = merged_df.groupby("Proverb")["Emotion"].apply(list)

# Υπολογίζουμε την πιθανότητα συμφωνίας μεταξύ των διαφορετικών ετικετών Emotion για κάθε παροιμία
def agreement_probability(emotions):
    if len(emotions) < 2:
        return 0

    total_pairs = 0
    agreeing_pairs = 0

    # Δημιουργούμε όλα τα δυνατά ζεύγη από ετικέτες Emotion
    for e1, e2 in combinations(emotions, 2):
        total_pairs += 1
        if e1 == e2:
            agreeing_pairs += 1

    # Υπολογίζουμε την πιθανότητα συμφωνίας αν υπάρχουν ζεύγη για σύγκριση
    return agreeing_pairs / total_pairs if total_pairs > 0 else 0

# Υπολογίζουμε την πιθανότητα συμφωνίας για κάθε παροιμία
agreement_probs = emotion_annotations.apply(agreement_probability)

# Παίρνουμε τις 3 παροιμίες με τη μεγαλύτερη πιθανότητα συμφωνίας
top_3_agreement = agreement_probs.nlargest(3)

# Εμφάνιση των αποτελεσμάτων
print("\nΟι 3 παροιμίες με τη μεγαλύτερη πιθανότητα συμφωνίας στο Emotion:")
print(top_3_agreement)

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Για κάθε Emotion, συλλέγουμε όλα τα μοναδικά κείμενα που έχουν χαρακτηριστεί έστω και μία φορά με αυτό το Emotion.
emotion_groups = merged_df.groupby("Emotion")["Proverb"].unique()

# Μετατρέπουμε σε dictionary για επεξεργασία
emotion_texts = {emotion: " ".join(texts) for emotion, texts in emotion_groups.items()}

# Μετατρέπουμε σε DataFrame
emotion_df = pd.DataFrame(list(emotion_texts.items()), columns=["Emotion", "Text"])

vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, min_df=2, max_df=0.55)

tfidf_matrix = vectorizer.fit_transform(emotion_df["Text"])

# Λήψη των λέξεων
feature_names = vectorizer.get_feature_names_out()

# Μετατροπή του TF-IDF πίνακα σε DataFrame για εύκολη ανάλυση
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=emotion_df["Emotion"], columns=feature_names)

# Βρίσκουμε τις κορυφαίες λέξεις ανά Emotion (με βάση το TF-IDF score)
top_terms_per_emotion = {}
for emotion in tfidf_df.index:
    top_terms = tfidf_df.loc[emotion].nlargest(10).index.tolist()  # Επιλογή των 10 σημαντικότερων όρων
    top_terms_per_emotion[emotion] = top_terms

# Εμφάνιση αποτελεσμάτων
print("\nΚορυφαίοι όροι TF-IDF ανά Emotion:")
for emotion, terms in top_terms_per_emotion.items():
    print(f"{emotion}: {', '.join(terms)}")