import os
import shutil
from pathlib import Path

def main():
    root = Path(r"D:\bcp1")
    project_root = root / "Fraud-Detection-Project" / "fraud_platform_imen"
    
    # 1. Setup Directories
    tesseract_dir = root / "Tesseract-OCR"
    archive_dir = project_root / "archive_unused"
    
    tesseract_dir.mkdir(exist_ok=True)
    archive_dir.mkdir(exist_ok=True)
    
    print("🧹 Starting PharmaCare Cleanup...\n")
    
    # 2. Cleanup Tesseract mess in D:\bcp1
    tess_exts = {".dll", ".exe", ".html"}
    moved_tess = 0
    if root.exists():
        for file in root.iterdir():
            if file.is_file() and file.suffix.lower() in tess_exts:
                try:
                    shutil.move(str(file), str(tesseract_dir / file.name))
                    moved_tess += 1
                except Exception as e:
                    print(f"  [Error moving {file.name}]: {e}")
        
        # Move specific folders
        for folder in ["tessdata", "doc"]:
            src = root / folder
            if src.exists() and src.is_dir():
                try:
                    shutil.move(str(src), str(tesseract_dir / folder))
                    moved_tess += 1
                except Exception as e:
                    pass
    
    print(f"✅ Moved {moved_tess} Tesseract files/folders to D:\\bcp1\\Tesseract-OCR")
    
    # 3. Archive dead project code
    archive_items = [
        "main_app_code/login_page.py",
        "main_app_code/20230829T163443Z-001.zip",
        "main_app_code/20230829T163443Z-001",
        "test.py",
        "test",
        "session_state.zip",
        "session_state",
        "widgets",
        "main_app_code/details.txt"
    ]
    
    archived_count = 0
    for item in archive_items:
        src = project_root / item
        if src.exists():
            try:
                # If archiving a folder with the same name, rename it to avoid conflicts
                dest_name = src.name
                shutil.move(str(src), str(archive_dir / dest_name))
                archived_count += 1
            except Exception as e:
                print(f"  [Error archiving {item}]: {e}")
                
    print(f"✅ Archived {archived_count} unused files/folders to {archive_dir.name}")
    
    # 4. Delete bad data
    deleted_count = 0
    delete_items = ["accounts.db", ".venv311"]
    for item in delete_items:
        src = project_root / item
        if src.exists():
            try:
                if src.is_dir():
                    shutil.rmtree(src)
                else:
                    src.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"  [Error deleting {item}]: {e}")
                
    print(f"✅ Deleted {deleted_count} obsolete files/folders (accounts.db, .venv311)")
    
    print("\n🎉 Cleanup Complete! The project is now clean.")

if __name__ == "__main__":
    main()
