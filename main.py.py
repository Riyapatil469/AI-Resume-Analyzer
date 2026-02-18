import PyPDF2
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 50)
print("        AI RESUME ANALYZER")
print("=" * 50)

# ---------------------------
# STEP 1: Get Resume PDF Path
# ---------------------------
resume_path = input("\nEnter resume file path (example: resume 3.pdf): ")

try:
    with open(resume_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        resume_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                resume_text += text
except FileNotFoundError:
    print("❌ File not found. Please check the file name.")
    exit()

# ---------------------------
# STEP 2: Paste Job Description
# ---------------------------
print("\nPaste Job Description (Press Enter twice when done):")

job_description = ""
while True:
    line = input()
    if line == "":
        break
    job_description += line + " "

# ---------------------------
# STEP 3: Clean Text
# ---------------------------
resume_text = resume_text.lower().translate(str.maketrans('', '', string.punctuation))
job_description = job_description.lower().translate(str.maketrans('', '', string.punctuation))

# ---------------------------
# STEP 4: TF-IDF Similarity
# ---------------------------
documents = [resume_text, job_description]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
match_score = round(similarity[0][0] * 100, 2)

# ---------------------------
# STEP 5: Display Result
# ---------------------------
print("\n---- AI Match Analysis ----")
print("Match Percentage:", match_score, "%")

if match_score > 75:
    print("Excellent match! ✅")
elif match_score > 50:
    print("Good match 👍")
else:
    print("Low match. Consider improving resume.")

