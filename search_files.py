import os

search_extensions = [".csv"]
keywords = ["Titanic", "train", "test"]

for root, dirs, files in os.walk("C:\\", topdown=True):
    for name in files:
        if any(name.lower().endswith(ext) for ext in search_extensions):
            if any(keyword in name.lower() for keyword in keywords):
                print(os.path.join(root, name))
