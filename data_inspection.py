from datasets import load_dataset
import matplotlib.pyplot as plt
import nltk
import re
from nltk.tokenize import word_tokenize
nltk.download('punkt')

print("📥 Loading Wikipedia dataset...")
dataset = load_dataset("not-lain/wikipedia")  # sample for demo

print("🔍 Dataset Info:")
print(dataset)
print(dataset["train"].info)
x = input("Press Enter to continue...")

print("🔍 Dataset Sample:")
print(dataset["train"][0])
print("Article length: ", len(dataset["train"][0]["text"].split(" ")))
x = input("Press Enter to continue...")

print("🎨 Plotting histogram...")
lengths = [len(word_tokenize(x)) for x in dataset["train"]["text"]]
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=100, color='skyblue', edgecolor='black')
plt.title("Distribution of Word Counts in Wikipedia Articles")
plt.xlabel("Word Count")
plt.ylabel("Number of Articles")
plt.grid(True)
plt.show()

print("Article with less than 100 words: ", len([x for x in lengths if x < 100]))
print("Article with less than 500 words: ", len([x for x in lengths if x < 500]))
print("Article with less than 1000 words: ", len([x for x in lengths if x < 1000]))

print("🔍 Inspecting small articles...")
x = input("Press Enter to continue...")
small_articles = dataset["train"].filter(lambda x: len(x['text'].split()) < 500)
for i, article in enumerate(small_articles):
    print(article["text"])
    print("Article length: ", len(word_tokenize(article["text"])))
    print("--------------------------------")
    x = input(f"[{i+1}/{len(small_articles)}]Press Enter to continue, or type 'q' to quit:")
    if x == "q":
        break

print("🔍 Inspecting articles that may refer to other articles...")
x = input("Press Enter to continue...")
refer_articles = dataset["train"].filter(
    lambda x: re.search(r"may also refer to:", x["text"], re.IGNORECASE) is not None
)
for i, article in enumerate(refer_articles):
    print(article["text"])
    print("Article length (words):", len(word_tokenize(article["text"])))
    print("--------------------------------")
    
    x = input(f"[{i+1}/{len(refer_articles)}] Press Enter to continue, or type 'q' to quit: ")
    if x.lower() == "q":
        break
