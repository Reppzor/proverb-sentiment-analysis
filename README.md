# Proverb Sentiment & Memeability Analysis

**This project was developed as part of the NLP course in the MSc in Digital Methods for the Humanities and Social Sciences at the University of Athens (Spring 2025).**

---

## 🧰 Tools & Libraries
- Python 3.10  
- Pandas  
- scikit-learn  
- Stanza (for Greek NLP)  
- NLTK  
- Matplotlib, Seaborn  
- TF-IDF (sklearn)  
- Cohen’s Kappa (inter-annotator agreement)  
- Google Colab  

---

## 📄 Dataset  
A collection of Greek proverbs annotated by multiple human annotators for:
- **Sentiment** 
- **Emotion**  
- **Memeability** (whether the proverb has viral/meme potential)

The annotators were volunteering students partaking in the course. Each annotation file was manually prepared and merged to create a unified dataset. Duplicates were detected and removed using cosine similarity over TF-IDF vectors.

---

## 🧪 Methodology

The analysis focused on:
- **Measuring annotator agreement** using Cohen’s Kappa and percentage agreement  
- **Cleaning and deduplicating** proverb data  
- **Highlighting strong emotional signals** using TF-IDF per emotion class  
- **Identifying memeable expressions** based on crowd-sourced scores  

Visualizing annotator agreement on neutral sentiment detection:
![download](https://github.com/user-attachments/assets/fc67e19d-7cdb-49ff-9c1a-f49a40288365)


---

## 📊 Sample Results

### 🧠 Top TF-IDF Terms per Emotion Category:

| Emotion               | Top Terms |
|-----------------------|-----------|
| έκπληξη / σοκ         | αρνιά, κατσίκια, παιδί, έχεις, αλλόκοτες, απρόβλεπτες |
| έμπνευση / κίνητρο    | αφού, ντροπή, πιο, κότα, αυγό, αγάλι |
| ένταση                | πίπτει, βροντά, δικό, είσαι, αλέστε, δώστε |
| αγάπη                 | έρωτας, πόνος, παντοτινή, τσορβάς, λογαριασμός |
| αμηχανία              | έσκασε, βελόνι, βρέξει, γυρίζω, γύρισμα |
| ανακούφιση            | θεός, άνθρωπος, βοήθεια, πόνος, μήτε, κοιτάς |
| ανεπάρκεια            | αλίμονο, βυζαίνει, σκύλος, στ, γελάει, γης |

---

### 🧩 Most Agreed-Upon Proverbs (Emotion):
| Proverb                                                             | Agreement |
|---------------------------------------------------------------------|-----------|
| Έλα παππού να σου δείξω τ’ αμπελοχώραφά σου.                        | 0.55      |
| Όλοι με χρυσά βελούδα, ποιος τα βόσκει τα γαϊδούρια;                | 0.55      |
| Βρήκαμε παπά, ας θάψουμε πέντε-έξι.                                 | 0.47      |

---

### 🔥 Top 10 Most Memeable Proverbs

| Proverb                                                                 | Memeability Score |
|-------------------------------------------------------------------------|-------------------|
| Στους δύο τρίτος δε χωρεί.                                              | 54.0              |
| Γιάννης κερνάει και Γιάννης πίνει.                                      | 45.0              |
| Όταν λείπει η γάτα, χορεύουν τα ποντίκια.                               | 41.0              |
| Όταν ψοφήσουν τ’ άλογα, τιμή έχουν τα γαϊδούρια!                        | 32.0              |
| Χέσε κώλε κι άφσε κι όλε.                                              | 31.0              |
| Ο παπουτσής ξυπόλητος κι ο ράφτης μπαλωμένος.                          | 31.0              |
| Κάθε πρώτη του μηνός, για δεσπότης, για φανός.                         | 30.0              |
| Τα μεταξωτά βρακιά θέλουν και επιδέξιους κώλους.                        | 29.0              |
| Μην καμαρώνεις στην αρχή προτού ιδείς το τέλος.                         | 29.0              |
| Για να γίνεις ηγούμενος, πρέπει να σε πηδήξει ο προηγούμενος.          | 29.0              |

---

## 🏁 How to Run

1. Upload the `.xlsx` annotation files to your Colab environment.
2. Run [`proverb-sentiment-analysis.py`](./proverb-sentiment-analysis.py) to:
   - Merge, clean, and filter the data
   - Compute agreement metrics
   - Extract top emotional terms using TF-IDF
   - Visualize agreement using heatmaps
3. Check console outputs for the analysis results.

---
