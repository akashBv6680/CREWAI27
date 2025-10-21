import os
import imaplib
import email
from email.header import decode_header
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

# --- Configuration ---
IMAP_SERVER = os.getenv("IMAP_SERVER")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")

# --- Setup the Local LLM ---
# Make sure Ollama is running on your machine
llm = Ollama(
    model="llama3",  # Or "mistral", etc.
    base_url="http://localhost:11434"
)

# --- Email Processing Functions ---

def clean_filename(filename):
    """Removes invalid characters from a filename."""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()

def check_and_process_emails():
    """
    Checks for unread emails, uses an LLM to classify if they are project requests,
    downloads attachments, and returns the project context.
    """
    print("Connecting to email server...")
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_APP_PASSWORD)
        mail.select("inbox")
    except Exception as e:
        print(f"Error connecting to email server: {e}")
        return None

    # Search for all unread emails
    status, messages = mail.search(None, "UNSEEN")
    if status != "OK":
        print("No new messages found.")
        mail.logout()
        return None

    email_ids = messages[0].split()
    print(f"Found {len(email_ids)} unread emails. Processing...")

    project_context = None

    for email_id in email_ids:
        status, msg_data = mail.fetch(email_id, "(RFC822)")
        if status != "OK":
            continue

        msg = email.message_from_bytes(msg_data[0][1])
        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding if encoding else "utf-8")

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode()
                        break
                    except:
                        continue
        else:
            try:
                body = msg.get_payload(decode=True).decode()
            except:
                body = "Could not decode email body."

        email_content = f"Subject: {subject}\n\nBody:\n{body}"

        # --- Use LLM to classify the email ---
        print(f"\nAnalyzing email with subject: '{subject}'")
        classification_agent = Agent(
            role='Email Classifier',
            goal='Accurately classify if an email is a new project request based on its content.',
            backstory='You are an AI assistant responsible for filtering incoming emails to identify potential new projects, problem statements, or data submissions. Your decision must be precise.',
            llm=llm,
            verbose=False
        )
        classification_task = Task(
            description=f"Analyze the following email content and determine if it represents a new project request, a problem statement, or contains a dataset for a new task. Respond with 'YES' if it is, and 'NO' if it is not.\n\n--- EMAIL CONTENT ---\n{email_content}",
            expected_output="A single word: 'YES' or 'NO'.",
            agent=classification_agent
        )
        
        # We create a temporary crew for this single task
        is_project = Crew(agents=[classification_agent], tasks=[classification_task]).kickoff()
        
        if "YES" in is_project.upper():
            print(f"Project found! Subject: {subject}")
            project_context = {"email_content": email_content, "attachments": []}

            # Download attachments
            for part in msg.walk():
                if part.get_content_maintype() == 'multipart' or part.get('Content-Disposition') is None:
                    continue
                
                filename = part.get_filename()
                if filename:
                    cleaned_filename = clean_filename(filename)
                    filepath = os.path.join(os.getcwd(), cleaned_filename)
                    with open(filepath, "wb") as f:
                        f.write(part.get_payload(decode=True))
                    print(f"Downloaded attachment: {cleaned_filename}")
                    project_context["attachments"].append(filepath)

            # Mark email as read to avoid re-processing
            mail.store(email_id, "+FLAGS", "\\Seen")
            break # Process one project at a time
        else:
            print(f"Email '{subject}' is not a project request. Marking as read and skipping.")
            mail.store(email_id, "+FLAGS", "\\Seen")


    mail.logout()
    return project_context

# --- Agent and Task Definitions (largely the same, with updated descriptions) ---

# All agent definitions (project_manager, data_acquirer, etc.) are the same as before.
# Let's just re-paste one for context.
project_manager = Agent(
    role='Project Manager',
    goal='Oversee the entire project from inception to reporting, starting from an email request.',
    backstory=(
        "You are a seasoned Project Manager... You are the central hub of communication."
    ),
    verbose=True, allow_delegation=True, llm=llm
)
data_acquirer = Agent(role='Data Acquisition Team Lead', goal='Gather, clean, and prepare data.', backstory="You lead a team of data engineers...", verbose=True, llm=llm)
model_trainer = Agent(role='Model Training Team Lead', goal='Select, train, and validate the best model.', backstory="You are an expert in various ML algorithms...", verbose=True, llm=llm)
deployment_specialist = Agent(role='Deployment & Ops Team Lead', goal='Deploy and monitor the model.', backstory="You are a DevOps and MLOps expert...", verbose=True, llm=llm)
business_reporter = Agent(role='Business & Reporting Team Lead', goal='Translate technical results into business reports.', backstory="You are the bridge between technical teams and the client...", verbose=True, llm=llm)


# --- UPDATED Tasks to handle email input ---

kickoff_task = Task(
    description=(
        "A new project has been initiated via email. Here is the content:\n\n"
        "--- EMAIL CONTENT ---\n{email_content}\n\n"
        "Your first job is to thoroughly analyze this request. Create a structured project brief in 'project_brief.txt'. "
        "This brief must summarize the project's goals, identify the core problem, and list any mentioned data sources or attached files ({attachments})."
    ),
    expected_output='A text file named project_brief.txt with a structured summary of the project request.',
    agent=project_manager
)

data_task = Task(
    description=(
        "Read the 'project_brief.txt'. If any files were attached to the initial email (listed in the brief), "
        "use them as the primary data source. If not, gather data based on the project description. "
        "Perform all necessary cleaning, preparation, and feature engineering. "
        "Output the result to 'clean_data.csv'."
    ),
    expected_output='A CSV file named clean_data.csv ready for model training.',
    agent=data_acquirer,
    context=[kickoff_task]
)

# The rest of the tasks (training_task, deployment_task, reporting_task) are the SAME as before,
# as they depend on the files created by the previous steps.
training_task = Task(description="Take 'clean_data.csv' and the 'project_brief.txt' to train and validate a model. Output 'model_performance_report.txt'.", expected_output="A text file with model performance metrics.", agent=model_trainer, context=[data_task])
deployment_task = Task(description="Read the performance report and create a deployment plan in 'deployment_plan.txt'.", expected_output="A detailed deployment plan text file.", agent=deployment_specialist, context=[training_task])
reporting_task = Task(description="Synthesize all documents ('project_brief.txt', 'model_performance_report.txt', 'deployment_plan.txt') into a final client-facing report named 'client_report.md'.", expected_output="A final client report in Markdown format.", agent=business_reporter, context=[deployment_task])


# --- Main Execution Block ---

def run_workflow():
    # 1. Check for new project emails
    project_data = check_and_process_emails()

    # 2. If a project is found, run the crew
    if project_data:
        print("\n--- INITIATING CREWAI WORKFLOW ---")
        
        ai_workflow_crew = Crew(
            agents=[project_manager, data_acquirer, model_trainer, deployment_specialist, business_reporter],
            tasks=[kickoff_task, data_task, training_task, deployment_task, reporting_task],
            process=Process.sequential,
            verbose=2
        )

        result = ai_workflow_crew.kickoff(inputs=project_data)

        print("\n\n########################")
        print("## AI Workflow execution complete!")
        print("## Final Result:")
        print(result)
        print("########################")
    else:
        print("\nNo new projects found in the inbox. Workflow finished.")

if __name__ == "__main__":
    run_workflow()
