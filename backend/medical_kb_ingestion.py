"""
Medical Knowledge Base Ingestion Script
One-time setup to populate ChromaDB with authoritative medical sources
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
import google.generativeai as genai
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import time

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in environment")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# Authoritative Medical Sources
MEDICAL_SOURCES = [
    {
        "name": "WHO - Lab Test Interpretation",
        "url": "https://www.who.int/health-topics/laboratory-testing",
        "topics": ["lab tests", "blood tests", "cholesterol", "glucose"]
    },
    {
        "name": "CDC - Vaccination Information",
        "url": "https://www.cdc.gov/vaccines/",
        "topics": ["vaccination", "immunization", "flu shot"]
    },
    {
        "name": "NIH - Vitamin D",
        "url": "https://ods.od.nih.gov/factsheets/VitaminD-HealthProfessional/",
        "topics": ["vitamin d", "deficiency", "supplements"]
    },
    {
        "name": "Mayo Clinic - Cholesterol Levels",
        "url": "https://www.mayoclinic.org/diseases-conditions/high-blood-cholesterol/",
        "topics": ["cholesterol", "LDL", "HDL", "triglycerides"]
    },
    {
        "name": "Cleveland Clinic - Blood Pressure",
        "url": "https://my.clevelandclinic.org/health/articles/blood-pressure",
        "topics": ["blood pressure", "hypertension", "cardiovascular"]
    }
]

# Curated medical knowledge snippets (simplified)
# In production, scrape or use medical APIs
MEDICAL_KNOWLEDGE = [
    {
        "source": "WHO",
        "topic": "Cholesterol",
        "content": """
        Normal cholesterol levels:
        - Total cholesterol: Less than 200 mg/dL is desirable
        - LDL (bad) cholesterol: Less than 100 mg/dL is optimal
        - HDL (good) cholesterol: 60 mg/dL and above is protective
        - Triglycerides: Less than 150 mg/dL is normal
        
        High cholesterol is a risk factor for heart disease and stroke.
        Lifestyle changes (diet, exercise) and medications can help manage cholesterol.
        """
    },
    {
        "source": "NIH - National Institutes of Health",
        "topic": "Vitamin D",
        "content": """
        Vitamin D levels:
        - Deficient: Less than 20 ng/mL
        - Insufficient: 20-30 ng/mL
        - Sufficient: 30-100 ng/mL
        - Potentially toxic: Above 100 ng/mL
        
        Vitamin D is essential for bone health, immune function, and overall wellbeing.
        Sources include sunlight exposure, fatty fish, fortified foods, and supplements.
        Low levels are common, especially in winter or for people with limited sun exposure.
        """
    },
    {
        "source": "CDC - Centers for Disease Control",
        "topic": "Influenza Vaccination",
        "content": """
        Flu vaccines:
        - Recommended annually for everyone 6 months and older
        - Best to get vaccinated before flu season (October-November)
        - Vaccine effectiveness varies (40-60% typically)
        - Common side effects: soreness at injection site, mild fever, muscle aches
        - Severe reactions are rare
        
        The flu vaccine cannot give you the flu.
        Even when the vaccine doesn't completely prevent illness, it can reduce severity.
        """
    },
    {
        "source": "Mayo Clinic",
        "topic": "Blood Glucose",
        "content": """
        Blood glucose (sugar) levels:
        - Normal fasting: 70-99 mg/dL
        - Prediabetes: 100-125 mg/dL
        - Diabetes: 126 mg/dL or higher on two separate tests
        - Normal after meals: Less than 140 mg/dL
        
        High blood sugar (hyperglycemia) can indicate diabetes or prediabetes.
        Low blood sugar (hypoglycemia) can cause shakiness, confusion, and fainting.
        Regular monitoring is important for people with diabetes.
        """
    },
    {
        "source": "Cleveland Clinic",
        "topic": "Blood Pressure",
        "content": """
        Blood pressure ranges (systolic/diastolic):
        - Normal: Less than 120/80 mm Hg
        - Elevated: 120-129/less than 80 mm Hg
        - Stage 1 Hypertension: 130-139/80-89 mm Hg
        - Stage 2 Hypertension: 140/90 mm Hg or higher
        - Hypertensive crisis: Higher than 180/120 mm Hg
        
        High blood pressure often has no symptoms but increases risk of heart attack and stroke.
        Lifestyle changes and medications can effectively control blood pressure.
        """
    },
    {
        "source": "WebMD",
        "topic": "Complete Blood Count (CBC)",
        "content": """
        CBC measures:
        - Red blood cells (RBC): Carry oxygen; normal 4.5-5.5 million cells/mcL
        - White blood cells (WBC): Fight infection; normal 4,500-11,000 cells/mcL
        - Hemoglobin: Oxygen-carrying protein; normal 14-18 g/dL (men), 12-16 g/dL (women)
        - Hematocrit: Percentage of blood volume that is RBCs; normal 40-54% (men), 36-48% (women)
        - Platelets: Help blood clot; normal 150,000-400,000/mcL
        
        Abnormal values can indicate anemia, infection, blood disorders, or other conditions.
        """
    },
    {
        "source": "MedlinePlus",
        "topic": "Thyroid Function Tests",
        "content": """
        Thyroid tests:
        - TSH (thyroid-stimulating hormone): 0.4-4.0 mIU/L is normal
        - Free T4: 0.8-1.8 ng/dL is normal
        - Free T3: 2.3-4.2 pg/mL is normal
        
        High TSH with low T4 suggests hypothyroidism (underactive thyroid).
        Low TSH with high T4 suggests hyperthyroidism (overactive thyroid).
        Thyroid hormones regulate metabolism, energy, and many body functions.
        """
    },
    {
        "source": "UpToDate Medical Reference",
        "topic": "Kidney Function",
        "content": """
        Kidney function tests:
        - Creatinine: 0.6-1.2 mg/dL is normal
        - BUN (blood urea nitrogen): 7-20 mg/dL is normal
        - eGFR (estimated glomerular filtration rate): Above 60 mL/min is normal
        
        These tests assess how well kidneys filter waste from blood.
        Elevated creatinine or BUN may indicate kidney problems.
        eGFR below 60 for 3+ months indicates chronic kidney disease.
        """
    },
    {
        "source": "NIH - National Library of Medicine",
        "topic": "Liver Function Tests",
        "content": """
        Liver enzyme tests:
        - ALT (alanine aminotransferase): 7-56 units/L is normal
        - AST (aspartate aminotransferase): 10-40 units/L is normal
        - ALP (alkaline phosphatase): 44-147 units/L is normal
        - Bilirubin: 0.1-1.2 mg/dL is normal
        
        Elevated liver enzymes can indicate liver damage, inflammation, or disease.
        Causes include hepatitis, fatty liver disease, alcohol use, or medications.
        """
    },
    {
        "source": "American Heart Association",
        "topic": "Heart Health Indicators",
        "content": """
        Key heart health markers:
        - Total cholesterol: Less than 200 mg/dL
        - Blood pressure: Less than 120/80 mm Hg
        - Fasting blood sugar: Less than 100 mg/dL
        - BMI: 18.5-24.9 is normal weight
        - Waist circumference: Less than 40 inches (men), 35 inches (women)
        
        Cardiovascular disease is the leading cause of death globally.
        Regular exercise, healthy diet, not smoking, and managing stress are key preventive measures.
        """
    }
]


def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Simple text chunking"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def ingest_medical_knowledge():
    """Ingest medical knowledge into ChromaDB"""
    
    # Initialize ChromaDB
    client = chromadb.Client(ChromaSettings(
        persist_directory=CHROMA_DIR,
        anonymized_telemetry=False
    ))
    
    collection = client.get_or_create_collection(
        name="medical_knowledge",
        metadata={"description": "Authoritative medical knowledge base"}
    )
    
    print("Ingesting medical knowledge into ChromaDB...")
    
    for idx, item in enumerate(MEDICAL_KNOWLEDGE):
        # Chunk content
        chunks = chunk_text(item['content'])
        
        print(f"\nProcessing: {item['source']} - {item['topic']}")
        print(f"  Chunks: {len(chunks)}")
        
        # Generate embeddings
        for i, chunk in enumerate(chunks):
            chunk_id = f"kb_{idx}_{i}"
            
            try:
                # Embed text
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=chunk,
                    task_type="retrieval_document"
                )
                embedding = result['embedding']
                
                # Add to collection
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "source_name": item['source'],
                        "topic": item['topic'],
                        "url": MEDICAL_SOURCES[idx % len(MEDICAL_SOURCES)]['url'] if idx < len(MEDICAL_SOURCES) else "",
                        "source_type": "medical_knowledge",
                        "chunk_index": i
                    }]
                )
                
                print(f"    ✓ Chunk {i+1}/{len(chunks)}")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
    
    print(f"\n✅ Ingestion complete! Total items in collection: {collection.count()}")


if __name__ == "__main__":
    print("=" * 60)
    print("KEEP Medical Knowledge Base Ingestion")
    print("=" * 60)
    
    # Check API key
    if GOOGLE_API_KEY == "your_api_key_here":
        print("\n⚠️  Please set your GOOGLE_API_KEY in the script or environment")
        exit(1)
    
    ingest_medical_knowledge()
    
    print("\n" + "=" * 60)
    print("Done! Medical knowledge base is ready for RAG queries.")
    print("=" * 60)
