import re
import uuid
from typing import Literal, Optional, List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# --- Configuration ---
# Using the model from rag_docs.py
METADATA_LLM_MODEL = "llama-3.3-70b-versatile" 
PAGE_SIZE_ESTIMATE = 2000  # Characters per page if no markers

# --- Heuristics ---
DOC_TYPE_KEYWORDS = {
    "صحيفة دعوى": ["صحيفة افتتاح", "عريضة دعوى", "طلب افتتاح", "صحيفة الدعوى"],
    "مذكرة دفاع": ["مذكرة بدفاع", "مذكرة دفاع", "مذكرة رد"],
    "حافظة مستندات": ["حافظة مستندات", "بيان مستندات"],
    "محضر جلسة": ["محضر جلسة", "جلسة يوم"],
    "حكم تمهيدي": ["حكم تمهيدي", "حكمت المحكمة"],
}

# --- Schemas ---
DocTypeEnum = Literal[
    "صحيفة دعوى",      # Statement of Claim
    "مذكرة دفاع",      # Defense Memo
    "مذكرة رد",        # Reply Memo
    "حافظة مستندات",   # Evidence Portfolio
    "محضر جلسة",       # Hearing Minutes
    "حكم تمهيدي",      # Preliminary Judgment
    "غير محدد"         # Unknown
]

PartyEnum = Literal[
    "المدعي",          # Plaintiff
    "المدعى عليه",     # Defendant
    "النيابة",         # Prosecution
    "المحكمة",         # The Court
    "خبير",            # Expert
    "غير محدد"         # Unknown/Neutral
]

class NormalizedChunk(BaseModel):
    """Atomic unit of text for the pipeline."""
    chunk_id: str = Field(..., description="Unique hash of the content")
    doc_id: str = Field(..., description="Parent document ID")
    
    # Traceability
    page_number: int = Field(..., description="1-based page number")
    paragraph_number: int = Field(..., description="1-based paragraph index on the page")
    
    # Content
    clean_text: str = Field(..., description="Text content with headers/stamps removed")
    
    # Context Metadata (Extracted in Node 0)
    doc_type: str = Field(..., description="Type of the parent document") # Typed as str to allow fallback but logically DocTypeEnum
    party: str = Field(..., description="The party submitting/authoring this document") # Typed as str

class Node0Output(BaseModel):
    chunks: List[NormalizedChunk]

class DocumentMetadata(BaseModel):
    doc_type: str = Field(description="The legal type of the document (e.g., صحيفة دعوى, مذكرة دفاع)")
    party: str = Field(description="The party presenting the document (e.g., المدعي, المدعى عليه)")

# --- Node 0 Class ---
class Node0_DocumentIntake:
    def __init__(self, llm=None):
        if llm:
            self.llm = llm
        else:
            self.llm = ChatGroq(model_name=METADATA_LLM_MODEL)
            
        self.metadata_parser = self.llm.with_structured_output(DocumentMetadata)
        
    def clean_text(self, text: str) -> str:
        # 1. Normalize Unicode
        text = text.replace("\u200f", "").replace("\u200e", "") # Remove directional marks
        
        # 2. Remove Tatweel (Kashida)
        text = re.sub(r"[ـ]+", "", text)
        
        # 3. Remove Page Numbers (e.g., - 12 - )
        text = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", text, flags=re.MULTILINE)
        
        # 4. Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def extract_metadata(self, header_text: str) -> DocumentMetadata:
        """
        Tries regex first, falls back to LLM.
        """
        # 1. Regex Heuristic
        found_type = "غير محدد"
        found_party = "غير محدد" # Default
        
        for dtype, keywords in DOC_TYPE_KEYWORDS.items():
            if any(k in header_text for k in keywords):
                found_type = dtype
                break
        
        # Heuristic for party is harder, but let's try basics
        if "المدعي" in header_text[:200]: # Very top usually says who
            found_party = "المدعي"
        elif "المدعى عليه" in header_text[:200]:
            found_party = "المدعى عليه"

        # If we have a confident type, we might skip LLM, but for robustness let's use LLM 
        # if found_type is unknown OR for better party extraction.
        # For this implementation, I'll prefer LLM if type is unknown, otherwise trust regex for type 
        # but maybe still check LLM for party if unknown.
        
        # Let's try LLM if type is unknown or party is unknown
        if found_type == "غير محدد" or found_party == "غير محدد":
             try:
                # 2. LLM Fallback
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """أنت نظام ذكي لتصنيف المستندات القانونية المصرية.
مهمتك: استخراج (نوع المستند) و (الصفة القانونية للجهة المقدمة) من النص التالي.

القواعد:
1. نوع المستند (doc_type) يجب أن يكون واحداً من: [صحيفة دعوى، مذكرة دفاع، مذكرة رد، حافظة مستندات، حكم تمهيدي، محضر جلسة].
2. الطرف (party) يجب أن يكون واحداً من: [المدعي، المدعى عليه، النيابة، المحكمة، خبير].
3. إذا كان النص غير واضح، ارجع "غير محدد".

النص:
{text}"""),
                ])
                
                # Using invoke with the formatted prompt
                formatted_prompt = prompt.invoke({"text": header_text[:2000]})
                result = self.metadata_parser.invoke(formatted_prompt)
                
                if found_type == "غير محدد":
                    found_type = result.doc_type
                if found_party == "غير محدد":
                    found_party = result.party
                    
             except Exception as e:
                 print(f"Metadata extraction LLM error: {e}")
                 # Fallback to defaults already set
        
        return DocumentMetadata(doc_type=found_type, party=found_party)

    def segment_document(self, raw_text: str, doc_id: str, metadata: DocumentMetadata) -> List[NormalizedChunk]:
        """
        Splits text into chunks with page/para tracking.
        """
        # To preserve paragraphs, we split by double newlines first.
        # But first we should normalize line endings.
        normalized_text = raw_text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split by double newlines (paragraphs)
        paras = re.split(r'\n\s*\n', normalized_text)
        
        chunks = []
        current_page = 1
        char_count = 0
        
        for idx, para in enumerate(paras):
            if not para.strip():
                continue
                
            # Clean the paragraph text individually
            cleaned_para = self.clean_text(para)
            if not cleaned_para:
                continue
                
            # Update page estimation (using length of ORIGINAL para to approximate page flow)
            char_count += len(para)
            if char_count > PAGE_SIZE_ESTIMATE:
                current_page += 1
                char_count = 0
            
            chunk = NormalizedChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                page_number=current_page,
                paragraph_number=idx + 1,
                clean_text=cleaned_para,
                party=metadata.party,
                doc_type=metadata.doc_type
            )
            chunks.append(chunk)
            
        return chunks

    def process(self, inputs: dict) -> dict:
        """
        Main entry point for LangGraph node.
        Input: {"raw_text": "...", "doc_id": "..."}
        """
        raw_text = inputs["raw_text"]
        doc_id = inputs.get("doc_id", "unknown")
        
        # 1. Identify Metadata (from first 2000 chars of RAW text)
        header_sample = raw_text[:2000]
        metadata = self.extract_metadata(header_sample)
        
        # 2. Segment and Clean
        chunks = self.segment_document(raw_text, doc_id, metadata)
            
        return {"chunks": chunks}
