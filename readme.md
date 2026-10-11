# ENTSO-E LGBM Forecaster
*🇬🇧 [English Version](#english-version) | 🇬🇷 [Ελληνική Έκδοση](#ελληνική-έκδοση)*

---

## 🇬🇧 English Version

This repository contains a pre-trained LightGBM (Machine Learning) model designed to forecast Day-Ahead electricity prices using the ENTSO-E API. The system fetches real-time data and generates predictions automatically at a 15-minute resolution.

The project is written in **Python** and is fully **Dockerized**, ensuring a consistent and isolated execution environment across any operating system.

### ⚙️ How It Works

1. **Early Morning Execution & Imputation:** Unlike standard models that wait for afternoon releases, this pipeline is engineered for **early morning execution**, prior to ENTSO-E gate closure and before cross-border data (like German `price_de_lu`) is published. 
2. **Smart Data Handling:** It uses a custom `smart_impute` engine over a 10-day rolling window. Missing future data is handled via strictly bounded interpolation (`limit_area='inside'`), 24-hour historical shifting (`shift(96)`), and **Linear Wind Blending** to ensure seamless midnight boundary transitions.
3. **Inference:** The system loads the highly optimized, pre-trained LightGBM model. It combines meteorological forecasts with historical autoregressive price lags (48h, 72h, and 168h) to generate forecasts instantly without retraining.
4. **Export:** The predicted values are automatically saved as a `.csv` file in the designated output directory.

### 📂 Repository Structure

* `main.py`: The main execution script. **Note:** This script dynamically handles the installation of required Python libraries natively, so no separate `requirements.txt` is needed.
* `entsoe_day_ahead_API.py`: The module responsible for interacting with the ENTSO-E REST API.
* `Dockerfile`: Instructions for building the isolated container environment.
* `[Pre-trained Model]`: The saved LightGBM model used for inference (e.g., `.pkl` or `.txt`).
* `LGBM Forecaster ENTSO-E.pdf`: Comprehensive documentation detailing the methodology, data processing, and model architecture.

### 🛠️ Prerequisites

To run this project, you only need to have **Docker** installed on your machine.
* [Get Docker](https://docs.docker.com/get-docker/)

### 🚀 Installation & Usage

Follow these steps in your terminal to generate today's forecast:

**1. Build the Docker Image:**
Build the environment using the provided Dockerfile.
```bash
docker build --no-cache -t lgbm-forecaster .
```

**2. Run the Container:**
Execute the model to fetch data and generate predictions.
```bash
docker run --rm -v "${PWD}:/app/data" -v "${PWD}:/app/output" lgbm-forecaster
```
*(The `-v` flags ensure that the script can access the local pre-trained model and save the resulting `.csv` forecast file back to your machine).*

---

## 🇬🇷 Ελληνική Έκδοση

Αυτό το project περιέχει ένα προεκπαιδευμένο μοντέλο μηχανικής μάθησης (LightGBM), το οποίο χρησιμοποιεί δεδομένα από το ENTSO-E API για να κάνει προβλέψεις τιμών ενέργειας Day-Ahead σε ανάλυση 15-λέπτου. Το σύστημα ανακτά δεδομένα σε πραγματικό χρόνο και παράγει τις προβλέψεις αυτόματα.

Είναι γραμμένο σε **Python** και πλήρως **Dockerized**, ώστε να εκτελείται εύκολα σε οποιονδήποτε υπολογιστή.

### ⚙️ Πώς Λειτουργεί

1. **Πρωινή Εκτέλεση & Αναπλήρωση:** Σε αντίθεση με τυπικά μοντέλα, το σύστημα είναι σχεδιασμένο για **πρωινή εκτέλεση**, πριν κλείσει η αγορά του ENTSO-E και πριν δημοσιευτούν τα διασυνοριακά δεδομένα (π.χ. οι γερμανικές τιμές `price_de_lu`). 
2. **Προηγμένη Διαχείριση Δεδομένων:** Ενσωματώνει τον μηχανισμό `smart_impute` σε κυλιόμενο παράθυρο 10 ημερών. Τα κενά στις μελλοντικές ώρες αναπληρώνονται δυναμικά μέσω εσωτερικής παρεμβολής (`limit_area='inside'`), ιστορικής μετατόπισης 24ώρου (`shift(96)`) και **Γραμμικής Μίξης Αιολικών (Linear Wind Blending)** για ομαλή μετάβαση τα μεσάνυχτα.
3. **Πρόβλεψη (Inference):** Το σύστημα φορτώνει το ήδη εκπαιδευμένο μοντέλο LightGBM συνδυάζοντας τις καιρικές προβλέψεις με ιστορικές υστερήσεις (lags 48, 72 και 168 ωρών) για άμεση παραγωγή αποτελεσμάτων χωρίς επανεκπαίδευση.
4. **Εξαγωγή:** Οι προβλεπόμενες τιμές αποθηκεύονται αυτόματα σε ένα ξεχωριστό αρχείο `.csv` στον φάκελο εξόδου.

### 📂 Δομή του Repository

* `main.py`: Ο κύριος κώδικας του συστήματος. **Σημείωση:** Το αρχείο αυτό αναλαμβάνει εσωτερικά και την εγκατάσταση των απαραίτητων βιβλιοθηκών Python, οπότε δεν απαιτείται αρχείο `requirements.txt`.
* `entsoe_day_ahead_API.py`: Το αρχείο που διαχειρίζεται την επικοινωνία με το ENTSO-E API.
* `Dockerfile`: Οι οδηγίες για τη δημιουργία του περιβάλλοντος Docker.
* `[Προεκπαιδευμένο Μοντέλο]`: Το έτοιμο μοντέλο LightGBM που χρησιμοποιείται για τις προβλέψεις.
* `LGBM Forecaster ENTSO-E.pdf`: Η πλήρης αναφορά υλοποίησης με λεπτομέρειες για τη μεθοδολογία και την αρχιτεκτονική.

### 🛠️ Προαπαιτούμενα

Το μόνο που χρειάζεται να έχετε εγκατεστημένο στον υπολογιστή σας είναι το **Docker**.
* [Λήψη Docker](https://docs.docker.com/get-docker/)

### 🚀 Οδηγίες Εγκατάστασης & Εκτέλεσης

Εκτελέστε τις παρακάτω εντολές στο τερματικό σας:

**1. Χτίσιμο (Build) του Image:**
```bash
docker build --no-cache -t lgbm-forecaster .
```

**2. Εκτέλεση (Run) του Container:**
```bash
docker run --rm -v "${PWD}:/app/data" -v "${PWD}:/app/output" lgbm-forecaster
```
*(Οι παράμετροι `-v` εξασφαλίζουν ότι το παραγόμενο αρχείο `.csv` με τις προβλέψεις θα αποθηκευτεί τοπικά στον υπολογιστή σας και ότι το script έχει πρόσβαση στο μοντέλο).*