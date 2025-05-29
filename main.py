import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import PyPDF2
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb
from chromadb.utils import embedding_functions
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models for structured output
class DocumentInfo(BaseModel):
    type: str = Field(description="Type of document (bill, discharge_summary, id_card)")
    hospital_name: Optional[str] = Field(description="Name of hospital/clinic")
    total_amount: Optional[float] = Field(description="Total amount from bill")
    date_of_service: Optional[str] = Field(description="Date of service")
    patient_name: Optional[str] = Field(description="Patient name")
    diagnosis: Optional[str] = Field(description="Medical diagnosis")
    admission_date: Optional[str] = Field(description="Admission date")
    discharge_date: Optional[str] = Field(description="Discharge date")
    policy_number: Optional[str] = Field(description="Insurance policy number")
    patient_id: Optional[str] = Field(description="Patient ID")

class ValidationResult(BaseModel):
    missing_documents: List[str] = Field(description="List of missing required documents")
    discrepancies: List[str] = Field(description="List of data discrepancies found")

class ClaimDecision(BaseModel):
    status: str = Field(description="approved or rejected")
    reason: str = Field(description="Reason for the decision")
    confidence_score: float = Field(description="Confidence score 0-1")

class ClaimProcessingResult(BaseModel):
    documents: List[DocumentInfo]
    validation: ValidationResult
    claim_decision: ClaimDecision

# Vector Database Setup
class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="medical_knowledge",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
    
    def add_medical_knowledge(self):
        """Add medical knowledge base for validation"""
        medical_knowledge = [
            {
                "id": "1",
                "content": "Medical bills must include hospital name, patient details, treatment dates, and itemized costs",
                "metadata": {"type": "validation_rule"}
            },
            {
                "id": "2", 
                "content": "Discharge summaries must contain patient name, diagnosis, admission/discharge dates, and treating physician",
                "metadata": {"type": "validation_rule"}
            },
            {
                "id": "3",
                "content": "Insurance claims require matching patient names across all documents and consistent dates",
                "metadata": {"type": "validation_rule"}
            }
        ]
        
        for item in medical_knowledge:
            self.collection.add(
                documents=[item["content"]],
                ids=[item["id"]],
                metadatas=[item["metadata"]]
            )
    
    def query_knowledge(self, query: str, n_results: int = 3):
        """Query medical knowledge base"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

# PDF Text Extraction
class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""

# Initialize LLM
def get_llm():
    """Initialize Gemini LLM"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.1
    )

# Agent Definitions
class MedicalClaimAgents:
    def __init__(self, llm):
        self.llm = llm
        self.vector_store = VectorStore()
        self.vector_store.add_medical_knowledge()
    
    def create_document_classifier_agent(self):
        """Agent to classify document types"""
        return Agent(
            role="Medical Document Classifier",
            goal="Accurately classify medical documents based on their content and filename",
            backstory="""You are an expert medical document analyst with years of experience 
            in healthcare administration. You can quickly identify document types like medical bills, 
            discharge summaries, insurance cards, and other medical documents based on their content 
            structure and key identifying features.""",
            llm=self.llm,
            verbose=True,
            memory=True,
        )
    
    def create_bill_processing_agent(self):
        """Agent specialized in processing medical bills"""
        return Agent(
            role="Medical Bill Processor",
            goal="Extract detailed billing information from medical bills including hospital details, amounts, and service dates",
            backstory="""You are a specialized medical billing expert who understands the complex 
            structure of hospital bills, insurance claims, and medical invoices. You can accurately 
            extract hospital names, patient information, itemized charges, total amounts, and service 
            dates from various billing document formats.""",
            llm=self.llm,
            verbose=True,
            memory=True,
        )
    
    def create_discharge_processing_agent(self):
        """Agent specialized in processing discharge summaries"""
        return Agent(
            role="Medical Discharge Summary Processor",
            goal="Extract critical patient information from discharge summaries including diagnosis, dates, and treatment details",
            backstory="""You are a medical records specialist with extensive experience in analyzing 
            discharge summaries and patient documentation. You excel at extracting patient names, 
            medical diagnoses, admission and discharge dates, treatment summaries, and physician 
            information from complex medical documents.""",
            llm=self.llm,
            verbose=True,
            memory=True,
        )
    
    def create_validation_agent(self):
        """Agent to validate extracted data and check for inconsistencies"""
        return Agent(
            role="Medical Claim Validator",
            goal="Validate extracted medical claim data for completeness, consistency, and accuracy",
            backstory="""You are a meticulous medical claim auditor with expertise in identifying 
            discrepancies, missing information, and inconsistencies in medical documentation. You 
            ensure all required documents are present and that patient information matches across 
            different documents.""",
            llm=self.llm,
            verbose=True,
            memory=True,
        )
    
    def create_decision_agent(self):
        """Agent to make final claim approval/rejection decision"""
        return Agent(
            role="Medical Claim Decision Maker",
            goal="Make informed decisions on medical claim approval or rejection based on validated data",
            backstory="""You are a senior medical claim adjudicator with years of experience in 
            insurance claim processing. You make fair and accurate decisions on claim approvals 
            based on policy requirements, document completeness, and data consistency. You always 
            provide clear reasoning for your decisions.""",
            llm=self.llm,
            verbose=True,
            memory=True,
        )

# Task Definitions
class MedicalClaimTasks:
    def __init__(self, agents: MedicalClaimAgents):
        self.agents = agents
    
    def create_classification_task(self, file_info: Dict[str, str]):
        """Task to classify document type"""
        return Task(
            description=f"""
            Classify the document type based on the filename '{file_info['filename']}' and 
            extracted text content:
            
            Text Content:
            {file_info['content'][:1000]}...
            
            Determine if this is a:
            - medical_bill
            - discharge_summary  
            - id_card
            - other
            
            Consider the document structure, headers, and key identifying elements.
            """,
            expected_output="Document type classification (medical_bill, discharge_summary, id_card, or other)",
            agent=self.agents.create_document_classifier_agent(),
        )
    
    def create_bill_processing_task(self, content: str):
        """Task to process medical bill"""
        return Task(
            description=f"""
            Extract detailed information from this medical bill:
            
            {content}
            
            Extract:
            - Hospital/clinic name
            - Total amount charged
            - Date of service
            - Patient name
            - Any policy/claim numbers
            - Key services provided
            
            Be precise with dates and amounts.
            """,
            expected_output="Structured billing information with hospital name, amounts, dates, and patient details",
            output_pydantic=DocumentInfo,
            agent=self.agents.create_bill_processing_agent(),
        )
    
    def create_discharge_processing_task(self, content: str):
        """Task to process discharge summary"""
        return Task(
            description=f"""
            Extract critical information from this discharge summary:
            
            {content}
            
            Extract:
            - Patient name
            - Primary diagnosis
            - Admission date
            - Discharge date
            - Hospital name
            - Treating physician (if available)
            - Key procedures or treatments
            
            Ensure dates are in consistent format (YYYY-MM-DD).
            """,
            expected_output="Structured discharge information with patient details, diagnosis, and dates",
            output_pydantic=DocumentInfo,
            agent=self.agents.create_discharge_processing_agent(),
        )
    
    def create_validation_task(self, documents: List[DocumentInfo]):
        """Task to validate extracted data"""
        return Task(
            description=f"""
            Validate the following extracted documents for a medical claim:
            
            {json.dumps([doc.dict() for doc in documents], indent=2)}
            
            Check for:
            1. Missing required documents (typically need bill and discharge summary)
            2. Patient name consistency across documents
            3. Date consistency (service dates should align with admission/discharge)
            4. Missing critical information in any document
            5. Unusual patterns or potential errors
            
            Use medical knowledge base to validate requirements.
            """,
            expected_output="Validation results with missing documents and discrepancies identified",
            output_pydantic=ValidationResult,
            agent=self.agents.create_validation_agent(),
        )
    
    def create_decision_task(self, documents: List[DocumentInfo], validation: ValidationResult):
        """Task to make claim decision"""
        return Task(
            description=f"""
            Make a claim decision based on:
            
            Documents:
            {json.dumps([doc.dict() for doc in documents], indent=2)}
            
            Validation Results:
            {validation.dict()}
            
            Decision criteria:
            - Approve if all required documents present and consistent
            - Reject if missing critical documents or major discrepancies
            - Consider document quality and completeness
            
            Provide confidence score (0-1) and clear reasoning.
            """,
            expected_output="Final claim decision with status, reason, and confidence score",
            output_pydantic=ClaimDecision,
            agent=self.agents.create_decision_agent(),
        )

# Main Claim Processor
class ClaimProcessor:
    def __init__(self):
        self.llm = get_llm()
        self.agents = MedicalClaimAgents(self.llm)
        self.tasks = MedicalClaimTasks(self.agents)
        self.pdf_processor = PDFProcessor()
    
    async def process_claim(self, file_paths: List[str]) -> ClaimProcessingResult:
        """Main method to process medical claim documents"""
        try:
            # Step 1: Extract text from PDFs
            logger.info("Extracting text from PDF files...")
            file_data = []
            for file_path in file_paths:
                content = self.pdf_processor.extract_text_from_pdf(file_path)
                file_data.append({
                    'filename': Path(file_path).name,
                    'content': content,
                    'path': file_path
                })
            
            # Step 2: Classify documents
            logger.info("Classifying documents...")
            classified_docs = []
            for file_info in file_data:
                classification_task = self.tasks.create_classification_task(file_info)
                crew = Crew(
                    agents=[self.agents.create_document_classifier_agent()],
                    tasks=[classification_task],
                    verbose=True
                )
                result = crew.kickoff()
                file_info['type'] = str(result).strip().lower()
                classified_docs.append(file_info)
            
            # Step 3: Process documents based on type
            logger.info("Processing documents by type...")
            processed_documents = []
            
            for doc in classified_docs:
                if 'bill' in doc['type']:
                    task = self.tasks.create_bill_processing_task(doc['content'])
                    crew = Crew(
                        agents=[self.agents.create_bill_processing_agent()],
                        tasks=[task],
                        verbose=True
                    )
                    result = crew.kickoff()
                    processed_documents.append(result)
                
                elif 'discharge' in doc['type']:
                    task = self.tasks.create_discharge_processing_task(doc['content'])
                    crew = Crew(
                        agents=[self.agents.create_discharge_processing_agent()],
                        tasks=[task],
                        verbose=True
                    )
                    result = crew.kickoff()
                    processed_documents.append(result)
            
            # Step 4: Validate extracted data
            logger.info("Validating extracted data...")
            validation_task = self.tasks.create_validation_task(processed_documents)
            crew = Crew(
                agents=[self.agents.create_validation_agent()],
                tasks=[validation_task],
                verbose=True
            )
            validation_result = crew.kickoff()
            
            # Step 5: Make final decision
            logger.info("Making claim decision...")
            decision_task = self.tasks.create_decision_task(processed_documents, validation_result)
            crew = Crew(
                agents=[self.agents.create_decision_agent()],
                tasks=[decision_task],
                verbose=True
            )
            decision_result = crew.kickoff()
            
            # Step 6: Compile final result
            final_result = ClaimProcessingResult(
                documents=processed_documents,
                validation=validation_result,
                claim_decision=decision_result
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing claim: {str(e)}")
            raise

# Example usage and testing
async def main():
    """Example usage of the claim processor"""
    # Initialize processor
    processor = ClaimProcessor()
    
    # Example file paths (replace with actual paths)
    file_paths = [
        "sample_bill.pdf",
        "sample_discharge.pdf",
        "sample_id_card.pdf"
    ]
    
    try:
        # Process claim
        result = await processor.process_claim(file_paths)
        
        # Print results
        print("\n" + "="*50)
        print("CLAIM PROCESSING RESULTS")
        print("="*50)
        print(json.dumps(result.dict(), indent=2, default=str))
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Set your Gemini API key as environment variable
    # os.environ["GEMINI_API_KEY"] = "your_api_key_here"
    
    asyncio.run(main())