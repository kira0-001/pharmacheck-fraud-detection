import os
import urllib.request
import subprocess
import sys

def main():
    print("🚀 Starting PharmaCare setup script...")
    
    # 1. Install pyspellchecker
    print("\n📦 Installing pyspellchecker...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyspellchecker"])
        print("✅ pyspellchecker installed successfully.")
    except Exception as e:
        print(f"❌ Failed to install pyspellchecker: {e}")

    # 2. Download language packs
    tessdata_dir = r"D:\bcp1\Tesseract-OCR\tessdata"
    
    if not os.path.exists(tessdata_dir):
        print(f"\n❌ Error: Cannot find tessdata directory at {tessdata_dir}")
        print("Make sure you ran the previous cleanup_project.py script first!")
        return
        
    languages = {
        "spa": "Spanish",
        "fra": "French",
        "deu": "German"
    }
    
    base_url = "https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/"
    
    print("\n🌍 Downloading Tesseract language packs...")
    for code, name in languages.items():
        filename = f"{code}.traineddata"
        filepath = os.path.join(tessdata_dir, filename)
        url = base_url + filename
        
        if os.path.exists(filepath):
            print(f"✅ {name} ({filename}) is already installed.")
        else:
            print(f"Downloading {name} ({filename})... this may take a minute (10-15MB)...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"✅ Downloaded {name} successfully!")
            except Exception as e:
                print(f"❌ Failed to download {name}: {e}")
                
    print("\n🎉 Setup complete! You can now restart your Streamlit server.")

if __name__ == "__main__":
    main()
