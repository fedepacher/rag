import logging
import smtplib
from email.mime.text import MIMEText
from fastapi import HTTPException


class Email:
    """
    Abstract class for email
    """
    def __init__(self):
        pass

    async def send(self, subject: str, content: str, email_destination: str):
        pass

    def receive(self):
        pass


class EmailSender(Email):

    def __init__(self, server: str, email: str, password: str, security_protocol: str):
        super().__init__()
        self.server = server
        self.email = email
        self.password = password
        self.security_protocol = security_protocol

    async def send(self, subject: str, content: str, email_destination: str):
        # Prepare the email
        email_body = content
        msg = MIMEText(email_body)
        msg['Subject'] = subject
        msg['From'] = self.email
        msg['To'] = email_destination

        # Send the email
        logging.info("Sending email with new password")
        try:
            # TLS
            if 'tls' in self.security_protocol.lower():
                with smtplib.SMTP(self.server, 587) as server:
                    server.starttls()
                    server.login(self.email, self.password)
                    server.sendmail(msg['From'], [msg['To']], msg.as_string())
            elif 'ssl' in self.security_protocol.lower():
                with smtplib.SMTP_SSL(self.server, 465) as server:
                    server.login( self.email, self.password)
                    server.sendmail(msg['From'], [msg['To']], msg.as_string())
        except smtplib.SMTPAuthenticationError as e:
            logging.error("SMTP authentication error: %s", e)
            raise HTTPException(status_code=500, detail="Failed to authenticate with the email server.")
        except Exception as e:
            logging.error("Failed to send email: %s", e)
            raise HTTPException(status_code=500, detail="Failed to send email")
