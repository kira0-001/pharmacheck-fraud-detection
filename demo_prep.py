import pandas as pd
import os

def prepare_demo_database():
    db_path = r"D:\bcp1\Fraud-Detection-Project\fraud_platform_imen\data\updated_ner_results.xlsx"
    print(f"Loading database from {db_path}...")
    
    try:
        df = pd.read_excel(db_path)
    except Exception as e:
        print(f"❌ Failed to load Excel file: {e}")
        return

    # Define the new authentic medicines for the live demo
    demo_drugs = [
        {
            "DRUGNAME": "PENTAB-20",
            "TYPE": "TABLETS",
            "COMPOSITION": "Pantoprazole Sodium Sesquihydrate Gastro-Resistant",
            "SIZE": "20 mg",
            "DOSAGE": "As directed by the physician."
        },
        {
            "DRUGNAME": "Health OK",
            "TYPE": "Tablets",
            "COMPOSITION": "Multivitamin, Multimineral, Amino Acids with Taurine and Ginseng Extract",
            "SIZE": "10 N",
            "DOSAGE": "One tablet daily"
        },
        {
            "DRUGNAME": "Calpol Tablets 650 mg",
            "TYPE": "Tablets",
            "COMPOSITION": "Paracetamol Tablets IP 650 mg",
            "SIZE": "15 Tablets",
            "DOSAGE": "Adults & children 12 years and above: 1 tablet 4-6 hourly upto maximum 4000mg per day."
        }
    ]

    # Convert to DataFrame
    new_records = pd.DataFrame(demo_drugs)
    
    # Check if they already exist to avoid duplicates
    if "PENTAB-20" in df['DRUGNAME'].values:
        print("✅ Demo drugs are already in the database!")
        return

    # Append the new records
    updated_df = pd.concat([df, new_records], ignore_index=True)
    
    # Save back to Excel
    updated_df.to_excel(db_path, index=False)
    print(f"🎉 Successfully added {len(demo_drugs)} new medicines to the database!")
    print(f"Total records is now: {len(updated_df)}")
    print("\n👉 Now restart Streamlit and scan your images again. They should turn GREEN!")

if __name__ == "__main__":
    prepare_demo_database()
