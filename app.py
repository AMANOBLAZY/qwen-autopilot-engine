"""
Qwen Autopilot Support Engine - FastAPI Backend
Handles API routing, schema validation, and LangGraph execution.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from agent import qwen_agent, llm
from datetime import datetime, timezone
from typing import Optional
import os
import smtplib
from email.mime.text import MIMEText
import uuid

# ==========================================
# 1. APP INITIALIZATION & SECURITY
# ==========================================
app = FastAPI(
    title="Qwen Autopilot Support Engine",
    description="Enterprise API for Autonomous Customer Operations & HITL Management",
    version="2.1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base URL of the customer-facing frontend, used to build a "resume this chat" link
# that gets embedded in manager-initiated follow-up emails. Override via .env if the
# frontend is served from a different host/port.
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://127.0.0.1:5500/index.html")


@app.on_event("startup")
async def _log_smtp_status():
    """Prints whether real email delivery is configured, so it's obvious at a glance during a demo."""
    if os.getenv("SMTP_HOST"):
        print(f"[Email] SMTP configured — follow-up emails will be sent via {os.getenv('SMTP_HOST')}.")
    else:
        print(
            "[Email] No SMTP_HOST configured — follow-up emails will be LOGGED ONLY (mock mode), "
            "not actually delivered. Set SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASSWORD/SMTP_FROM "
            "in a .env file to send real emails."
        )


# In-memory store to prevent duplicate ticket processing for the same order
active_tickets = {}

# In-memory record of tickets a manager has already approved/rejected, so the customer-facing
# /ticket_status poll can show the final outcome even though execute_refund_node doesn't
# write directly into chat_history.
resolved_tickets = {}

# ==========================================
# 1b. MANAGER REVIEW QUEUE (Master-Detail Dashboard)
# ==========================================
# Global in-memory queue of tickets currently awaiting human review.
# Each entry is a lightweight summary (for the sidebar list) plus the full
# state snapshot (for the detail pane) so the frontend doesn't need a second round-trip.
manager_queue = []


def _queue_entry(ticket_id: str, state_values: dict) -> dict:
    """Builds a manager_queue entry from a LangGraph state snapshot."""
    history = state_values.get("chat_history", [])
    return {
        "ticket_id": ticket_id,
        "order_id": state_values.get("order_id") or "N/A",
        "ticket_category": state_values.get("ticket_category", "Other"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Lets the sidebar flag "the customer just replied" vs. "still waiting on the customer".
        "last_message_role": history[-1]["role"] if history else None,
        "details": state_values,
    }


def _upsert_manager_queue(ticket_id: str, state_values: dict) -> None:
    """Adds or refreshes a ticket's entry in the manager review queue."""
    global manager_queue
    manager_queue = [t for t in manager_queue if t["ticket_id"] != ticket_id]
    manager_queue.append(_queue_entry(ticket_id, state_values))


def _remove_from_manager_queue(ticket_id: str) -> None:
    """Removes a ticket from the manager review queue (resolved/rejected/dequeued)."""
    global manager_queue
    manager_queue = [t for t in manager_queue if t["ticket_id"] != ticket_id]


# ==========================================
# 1c. MANAGER-INITIATED "REQUEST MORE INFO" EMAILS
# ==========================================
# In-memory audit log of every follow-up email the AI has sent on a manager's behalf.
sent_emails_log = []


def _draft_followup_email(details: dict, manager_note: str) -> str:
    """Uses the Qwen LLM to draft a short, context-aware follow-up email to the customer."""
    prompt = f"""
    You are a professional enterprise customer support agent writing a follow-up email to a customer
    whose ticket is currently under manager review.

    Ticket category: {details.get('ticket_category', 'Other')}
    Order ID: {details.get('order_id') or 'N/A'}
    Customer's most recent message: "{details.get('ticket_text', '')}"
    What the manager specifically needs clarified: "{manager_note.strip() or 'General additional detail is needed to continue reviewing this ticket.'}"

    INSTRUCTIONS:
    - Write a short, polite, professional email (3-5 sentences).
    - Clearly and specifically ask for the information the manager needs.
    - Reference their ticket/order where relevant so the email feels personalized, not generic.
    - Sign off as "The Support Team".
    - Output ONLY the email body text, no subject line, no markdown.
    """
    response = llm.invoke(prompt)
    return response.content.strip()


def _send_email(to_address: str, subject: str, body: str) -> str:
    """
    Sends the follow-up email.
    Uses real SMTP if configured via environment variables (SMTP_HOST, SMTP_PORT,
    SMTP_USER, SMTP_PASSWORD, SMTP_FROM). Otherwise falls back to a logged mock-send
    so the workflow is fully demoable without live mail credentials.
    Returns the delivery mode used, for transparency in the API response.
    """
    smtp_host = os.getenv("SMTP_HOST")
    delivery_mode = "mock_logged"

    if smtp_host:
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = os.getenv("SMTP_FROM", "support@company.com")
            msg["To"] = to_address

            with smtplib.SMTP(smtp_host, int(os.getenv("SMTP_PORT", "587"))) as server:
                server.starttls()
                smtp_user = os.getenv("SMTP_USER")
                smtp_password = os.getenv("SMTP_PASSWORD")
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                server.sendmail(msg["From"], [to_address], msg.as_string())
            delivery_mode = "sent_via_smtp"
        except Exception as e:
            print(f"[Email] SMTP send failed, falling back to mock log: {e}")
            delivery_mode = "mock_fallback"

    if delivery_mode != "sent_via_smtp":
        print(f"\n[Email][MOCK SEND]\nTo: {to_address}\nSubject: {subject}\n\n{body}\n")

    sent_emails_log.append({
        "to": to_address,
        "subject": subject,
        "body": body,
        "delivery_mode": delivery_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return delivery_mode


# ==========================================
# 2. PYDANTIC SCHEMAS (VALIDATION)
# ==========================================
class TicketRequest(BaseModel):
    """Schema for the initial ticket submission."""
    user_email: str = Field(..., description="Customer's email address")
    user_phone: str = Field(default="", description="Optional contact number")

    # Strict regex validation: Empty string OR exactly 2 letters + 4 digits
    order_id: str = Field(
        default="",
        pattern=r"^(?:[A-Za-z]{2}[0-9]{4})?$",
        description="Optional formatted Order ID (e.g., AB1234)"
    )

    ticket_text: str = Field(..., min_length=5, description="The customer's message")
    has_attachment: bool = Field(default=False, description="True if visual proof is provided")
    attachment_data: Optional[str] = Field(default=None, description="Base64-encoded file content (no data: prefix)")
    attachment_filename: Optional[str] = Field(default=None, description="Original filename of the upload")
    attachment_mime: Optional[str] = Field(default=None, description="MIME type of the upload, e.g. image/png")


class CustomerReplyRequest(BaseModel):
    """Schema for continuous live-chat follow-ups."""
    ticket_id: str = Field(..., description="The LangGraph thread ID")
    reply_text: str = Field(..., min_length=1, description="The customer's chat reply")
    attachment_data: Optional[str] = Field(default=None, description="Base64-encoded file content (no data: prefix)")
    attachment_filename: Optional[str] = Field(default=None, description="Original filename of the upload")
    attachment_mime: Optional[str] = Field(default=None, description="MIME type of the upload, e.g. image/png")


class ApprovalRequest(BaseModel):
    """Schema for Manager Dashboard HITL decisions."""
    ticket_id: str = Field(..., description="The LangGraph thread ID waiting for approval")
    approve: bool = Field(..., description="True to execute action, False to reject")


class RequestInfoEmailRequest(BaseModel):
    """Schema for a manager asking the AI to email the customer for more detail."""
    ticket_id: str = Field(..., description="The LangGraph thread ID under review")
    manager_note: str = Field(default="", description="Optional specifics on what the manager needs clarified")


# ==========================================
# 3. API ENDPOINTS
# ==========================================

@app.post("/submit_ticket")
async def submit_ticket(req: TicketRequest):
    """
    Initializes a new LangGraph thread and processes the first customer message.
    """
    # 1. Deduplication Check (Locking by Email + Order ID)
    ticket_key = f"{req.user_email}_{req.order_id}"

    if ticket_key in active_tickets:
        return {
            "status": "Error",
            "message": f"You already have an active ticket processing for order {req.order_id}. Please wait for a response."
        }

    # 2. Generate Thread ID and Lock
    ticket_id = str(uuid.uuid4())
    active_tickets[ticket_key] = ticket_id

    # 3. Prepare State
    config = {"configurable": {"thread_id": ticket_id}}
    initial_state = {
        "ticket_id": ticket_id,
        "user_email": req.user_email,
        "user_phone": req.user_phone,
        "ticket_text": req.ticket_text,
        "order_id": req.order_id if req.order_id else None,
        "has_attachment": req.has_attachment,
        "sentiment_score": 0.0,
        "fraud_flag": False,
        "ticket_category": "",  # Will be populated by AI
        "human_required": False,
        "verification_passed": False,
        "agent_reply": "",
        "chat_history": [],
        "damage_complaint": False,
        "proof_verified": False,
        "attachment_data": req.attachment_data,
        "attachment_filename": req.attachment_filename,
        "attachment_mime": req.attachment_mime,
    }

    try:
        # 4. Execute Graph
        for _ in qwen_agent.stream(initial_state, config):
            pass

        current_state = qwen_agent.get_state(config)

        # 5. Handle Interruptions (HITL)
        if current_state.next:
            _upsert_manager_queue(ticket_id, current_state.values)
            return {
                "ticket_id": ticket_id,
                "status": "Awaiting Human Review",
                "details": current_state.values,
                "chat_history": current_state.values.get("chat_history", [])
            }

        # 6. Handle Autonomous Completion (Release Lock)
        if ticket_key in active_tickets:
            del active_tickets[ticket_key]

        return {
            "ticket_id": ticket_id,
            "status": "Autonomous Reply",
            "message": current_state.values.get("agent_reply", "System Error."),
            "chat_history": current_state.values.get("chat_history", [])
        }

    except Exception as e:
        if ticket_key in active_tickets:
            del active_tickets[ticket_key]
        raise HTTPException(status_code=500, detail=f"Graph Execution Failed: {str(e)}")


@app.post("/customer_reply")
async def customer_reply(req: CustomerReplyRequest):
    """
    Resumes an existing LangGraph thread to process follow-up chat messages.
    """
    config = {"configurable": {"thread_id": req.ticket_id}}

    # 1. Verify thread exists
    current_state = qwen_agent.get_state(config)
    if not current_state.values:
        raise HTTPException(status_code=404, detail="Ticket session expired or not found.")

    # 2. Inject new message into state
    update_data = {"ticket_text": req.reply_text}

    # Only overwrite attachment fields if the customer actually attached something in this reply,
    # so an existing attachment from an earlier turn isn't wiped out by a text-only follow-up.
    if req.attachment_data:
        update_data["attachment_data"] = req.attachment_data
        update_data["attachment_filename"] = req.attachment_filename
        update_data["attachment_mime"] = req.attachment_mime
        update_data["has_attachment"] = True

    try:
        # 3. Resume Graph execution
        for _ in qwen_agent.stream(update_data, config):
            pass

        final_state = qwen_agent.get_state(config)

        # 4. Keep the manager queue in sync in case this reply triggered a new escalation
        if final_state.next:
            _upsert_manager_queue(req.ticket_id, final_state.values)
        else:
            _remove_from_manager_queue(req.ticket_id)

        return {
            "status": "Awaiting Human Review" if final_state.next else "Autonomous Reply",
            "message": final_state.values.get("agent_reply", "System processing..."),
            "chat_history": final_state.values.get("chat_history", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat Execution Failed: {str(e)}")


@app.post("/approve_ticket")
async def approve_ticket(req: ApprovalRequest):
    """
    Resumes a halted graph based on human manager decision.
    """
    config = {"configurable": {"thread_id": req.ticket_id}}
    current_state = qwen_agent.get_state(config)

    # 1. Validate Interruption
    if not current_state.next:
        raise HTTPException(status_code=400, detail="No pending human approvals found for this ticket.")

    email = current_state.values.get("user_email", "")
    order = current_state.values.get("order_id", "")
    ticket_key = f"{email}_{order}"

    try:
        if req.approve:
            # 2a. Resume Graph to execute action
            for _ in qwen_agent.stream(None, config):
                pass
            final_state = qwen_agent.get_state(config)

            if ticket_key in active_tickets:
                del active_tickets[ticket_key]

            _remove_from_manager_queue(req.ticket_id)

            final_message = final_state.values.get("agent_reply", "Approved.")
            resolved_tickets[req.ticket_id] = {"status": "Resolved", "message": final_message}

            return {"status": "Resolved", "message": final_message}
        else:
            # 2b. Reject and cleanup
            if ticket_key in active_tickets:
                del active_tickets[ticket_key]

            _remove_from_manager_queue(req.ticket_id)

            final_message = "Your ticket was reviewed by a manager and could not be approved at this time."
            resolved_tickets[req.ticket_id] = {"status": "Rejected", "message": final_message}

            return {"status": "Rejected", "message": "Ticket manually rejected or acknowledged by manager."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Approval Execution Failed: {str(e)}")


@app.get("/ticket_status/{ticket_id}")
async def get_ticket_status(ticket_id: str):
    """
    Lets the customer-facing chat widget poll for updates while a ticket is under manager
    review — this is what surfaces a manager's "request more info" email in the customer's
    live chat, and what tells the customer once a manager has approved/rejected their ticket.
    """
    # 1. Already resolved by a manager — report the final outcome.
    if ticket_id in resolved_tickets:
        config = {"configurable": {"thread_id": ticket_id}}
        current_state = qwen_agent.get_state(config)
        history = list(current_state.values.get("chat_history", [])) if current_state.values else []
        resolution = resolved_tickets[ticket_id]
        # Surface the resolution as a final chat message without mutating the underlying graph state.
        history.append({"role": "agent", "content": resolution["message"]})
        return {
            "ticket_id": ticket_id,
            "status": resolution["status"],
            "message": resolution["message"],
            "chat_history": history
        }

    # 2. Still an active thread — report its current standing.
    config = {"configurable": {"thread_id": ticket_id}}
    current_state = qwen_agent.get_state(config)
    if not current_state.values:
        raise HTTPException(status_code=404, detail="Ticket session expired or not found.")

    return {
        "ticket_id": ticket_id,
        "status": "Awaiting Human Review" if current_state.next else "Active",
        "message": current_state.values.get("agent_reply", ""),
        "chat_history": current_state.values.get("chat_history", [])
    }


@app.post("/request_more_info")
async def request_more_info(req: RequestInfoEmailRequest):
    """
    Lets a manager ask the AI to draft and send a follow-up email to the customer
    requesting additional detail, WITHOUT resolving or dequeuing the ticket.
    The exchange is logged into the ticket's chat history so the AI (and the next
    manager who opens this ticket) has full context if the customer replies.
    """
    config = {"configurable": {"thread_id": req.ticket_id}}
    current_state = qwen_agent.get_state(config)

    if not current_state.values:
        raise HTTPException(status_code=404, detail="Ticket session expired or not found.")

    details = current_state.values
    customer_email = details.get("user_email")
    if not customer_email:
        raise HTTPException(status_code=400, detail="Ticket has no customer email on file.")

    try:
        ai_drafted_body = _draft_followup_email(details, req.manager_note)
        resume_link = f"{FRONTEND_BASE_URL}?resume={req.ticket_id}"
        email_body = (
            f"{ai_drafted_body}\n\n"
            f"Continue this conversation any time here: {resume_link}"
        )

        subject = f"Following up on your {details.get('ticket_category', 'support')} request"
        if details.get("order_id"):
            subject += f" (Order {details['order_id']})"

        delivery_mode = _send_email(customer_email, subject, email_body)

        # Log the outreach into the ticket's own chat history for continuity, so it shows up
        # instantly in the customer's live chat too if they're still on the page.
        history = details.get("chat_history", [])
        log_note = f"[Manager requested more info via email — sent to {customer_email}]\n{ai_drafted_body}"
        history.append({"role": "agent", "content": log_note})
        qwen_agent.update_state(config, {"chat_history": history})

        # Refresh the queue snapshot so the sidebar/detail pane reflect the updated history.
        refreshed_state = qwen_agent.get_state(config)
        _upsert_manager_queue(req.ticket_id, refreshed_state.values)

        return {
            "status": "Email Sent",
            "sent_to": customer_email,
            "subject": subject,
            "email_preview": email_body,
            "delivery_mode": delivery_mode,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send follow-up email: {str(e)}")


@app.get("/sent_emails")
async def get_sent_emails():
    """Audit log of every AI-drafted follow-up email sent on a manager's behalf (demo/debug view)."""
    sorted_log = sorted(sent_emails_log, key=lambda e: e["timestamp"], reverse=True)
    return {"count": len(sorted_log), "emails": sorted_log}


@app.get("/manager_queue")
async def get_manager_queue():
    """
    Returns the full list of tickets currently awaiting human review.
    Powers the Master (sidebar list) side of the Manager Dashboard's Master-Detail view.
    Sorted newest-first so the most recently escalated ticket appears on top.
    """
    sorted_queue = sorted(manager_queue, key=lambda t: t["timestamp"], reverse=True)
    return {"count": len(sorted_queue), "queue": sorted_queue}


@app.get("/manager_queue/{ticket_id}")
async def get_manager_queue_ticket(ticket_id: str):
    """Returns a single queued ticket's full detail (Detail side of the split view)."""
    for entry in manager_queue:
        if entry["ticket_id"] == ticket_id:
            return entry
    raise HTTPException(status_code=404, detail="Ticket not found in the manager queue.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)