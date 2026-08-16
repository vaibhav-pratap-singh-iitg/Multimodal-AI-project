# 1. Install required packages
!pip install -q wilds transformers torch torchvision nltk datasets

import torch
import nltk
from datasets import load_dataset

# 2. Download and load the full dataset directly from Hugging Face
print("Attempting to load Waterbirds dataset from Hugging Face...")
try:
    # Combine train, validation, and test into a single dataset
    dataset = load_dataset("grodino/waterbirds", split="train+validation+test")

    print(f"\nSuccess!")
    print(f"Total samples in combined dataset: {len(dataset)}")

except Exception as e:
    print(f"\nError during download: {e}")

# next step

# 1. Install required packages (swapped wilds for datasets)
!pip install -q transformers torch torchvision nltk datasets

import torch
import re
import nltk
import json
import os
from collections import Counter
from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.corpus import stopwords
from nltk.util import ngrams

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# 2. Load Waterbirds directly from Hugging Face
print("Loading Waterbirds dataset from Hugging Face...")
# Combining train + validation + test directly in the load step
dataset = load_dataset("grodino/waterbirds", split="train+validation+test")
total_images = len(dataset)
print(f"Total images across train+val+test: {total_images}")

# 3. Initialize BLIP model and processor
print("Initializing BLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
model.eval()

# 4. Run captioning on ALL images, in batches, with checkpointing
BATCH_SIZE = 32          # increase if you have GPU headroom, decrease if you hit OOM
CHECKPOINT_EVERY = 500   # images
checkpoint_path = "captions_checkpoint.json"

captions = []
start_idx = 0

# resume from checkpoint if one exists (protects against Colab disconnects)
if os.path.exists(checkpoint_path):
    with open(checkpoint_path) as f:
        saved = json.load(f)
    captions = saved["captions"]
    start_idx = saved["next_idx"]
    print(f"Resuming from checkpoint at image {start_idx}")

print("Running captioning on full dataset (this will take a while)...")
with torch.no_grad():
    for batch_start in range(start_idx, total_images, BATCH_SIZE):
        # Slice the dataset directly to get the batch of PIL images
        batch_images = dataset[batch_start : batch_start + BATCH_SIZE]["image"]

        # Ensure all images are in RGB format for BLIP
        images = [img.convert("RGB") for img in batch_images]

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        out = model.generate(**inputs, max_new_tokens=20)
        decoded = processor.batch_decode(out, skip_special_tokens=True)
        captions.extend(decoded)

        done = batch_start + len(images)
        if done % CHECKPOINT_EVERY < BATCH_SIZE:
            with open(checkpoint_path, "w") as f:
                json.dump({"captions": captions, "next_idx": done}, f)
            print(f"Processed {done}/{total_images} images... (checkpoint saved)")

print(f"\nCaptioning complete. Total captions: {len(captions)}")
print("Sample captions:")
for cap in captions[:5]:
    print("-", cap)

# 5. Extract frequent n-grams / keywords (uni, bi, AND trigrams)
print("\nExtracting keywords...")
stop_words = set(stopwords.words('english'))
all_phrases = []

for cap in captions:
    words = re.findall(r'\b[a-z]+\b', cap.lower())
    words_no_stop = [w for w in words if w not in stop_words]

    all_phrases.extend(words_no_stop)                                   # unigrams
    all_phrases.extend([" ".join(b) for b in ngrams(words_no_stop, 2)]) # bigrams
    all_phrases.extend([" ".join(t) for t in ngrams(words, 3)])         # trigrams

# Get top 500
freq = Counter(all_phrases)
extracted_keywords = [phrase for phrase, count in freq.most_common(500) if count >= 3]

# 6. Merge with generic handwritten list, dedupe, and write to file
handwritten_list = [
    "water", "land", "ocean", "beach", "forest", "tree", "branch", "leaf",
    "bird", "seabird", "duck", "gull", "lake", "grass", "mountain",
    "snow", "sky", "blue sky", "cloud", "dirt", "mud", "sand",
    "reflection", "blur", "bright", "dark", "wetland", "marsh", "pond",
    "river", "shoreline", "cliff", "rock", "field", "meadow", "pier",
    "dock", "nest", "flying", "swimming", "perched", "wading"
]

merged_set = set(extracted_keywords + handwritten_list)
final_wordbank = sorted([word.strip() for word in merged_set if len(word.strip()) > 1])

output_file = "wordbank.txt"
with open(output_file, "w") as f:
    for word in final_wordbank:
        f.write(word + "\n")

print(f"\nSuccessfully wrote {len(final_wordbank)} keywords to {output_file}.")
print("Preview of wordbank.txt:")
print(final_wordbank[:20])


# filtering

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords

# ADDED: punkt_tab and averaged_perceptron_tagger_eng to fix recent Colab LookupErrors
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

with open("wordbank.txt") as f:
    phrases = [line.strip() for line in f if line.strip()]

print(f"Starting with {len(phrases)} phrases")

# Words that describe "there is a bird" rather than background/context.
bird_words = {
    "bird", "birds", "wing", "wings", "beak", "feather", "feathers",
    "perched", "perching", "flying", "flies", "fly", "sitting", "sits",
    "standing", "stands", "swimming", "swims", "seabird", "seabirds",
    "duck", "gull" # added the specific bird types from your handwritten list
}

# Words with no descriptive content on their own
filler_words = {
    "a", "an", "the", "is", "are", "in", "on", "of", "with", "and",
    "this", "that", "it", "its", "there", "which", "who", "picture",
    "image", "photo", "photograph", "close", "up"
}

stop_words = set(stopwords.words('english'))

def clean_phrase(phrase):
    words = phrase.lower().split()
    # Remove bird-words and pure filler
    kept = [w for w in words if w not in bird_words and w not in filler_words]
    return " ".join(kept).strip()

def has_real_content(phrase):
    """Keep only phrases that still contain a noun or adjective after cleaning."""
    if not phrase or len(phrase) < 2:
        return False

    tokens = word_tokenize(phrase)

    # Drop if EVERY remaining word is a stopword (nothing substantive left)
    if all(w in stop_words for w in tokens):
        return False

    tagged = pos_tag(tokens)

    # NN, NNS, NNP, NNPS = nouns | JJ, JJR, JJS = adjectives
    # Included adjectives so you don't lose words like "bright", "dark", or "wet"
    valid_content = [w for w, tag in tagged if tag.startswith("NN") or tag.startswith("JJ")]

    if not valid_content:
        return False

    return True

cleaned = set()
for p in phrases:
    c = clean_phrase(p)
    if has_real_content(c):
        cleaned.add(c)

final = sorted(cleaned)
print(f"After cleaning: {len(final)} phrases")

with open("wordbank.txt", "w") as f:
    for p in final:
        f.write(p + "\n")

print("Preview:")
print(final[:30])


# after this final manual cleanup of wordbank.txt
