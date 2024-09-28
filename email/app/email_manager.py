import os
import smtplib
import imaplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import decode_header

from config import Config

from datetime import datetime



class EmailManager:
    def __init__(self):
        config = Config()
        # Cargar configuración
        self.db_name = config.db_name
        self.db_user = config.db_user
        self.db_pass = config.db_pass  # Definir en 'config' o usar una variable por defecto
        self.db_host = config.db_host  # Definir en 'config' o usar una variable por defecto
        self.db_port = config.db_port  # Definir en 'config' o usar una variable por defecto
        self.access_token_expire_minutes = config.access_token_expire_minutes  # Definir en 'config' o usar una variable por defecto
        self.secret_key = config.secret_key  # Definir en 'config' o usar una variable por defecto

        self.mongo_host = config.mongo_host  # Definir en 'config' o usar una variable por defecto
        self.mongo_db_name = config.mongo_db_name  # Definir en 'config' o usar una variable por defecto
        self.mongo_port = config.mongo_port  # Definir en 'config' o usar una variable por defecto
        self.mongo_user = config.mongo_user  # Definir en 'config' o usar una variable por defecto
        self.mongo_pass = config.mongo_pass  # Definir en 'config' o usar una variable por defecto

        self.smtp_server = config.smtp_server  # Definir en 'config' o usar una variable por defecto
        self.smtp_port = config.smtp_port  # Definir en 'config' o usar una variable por defecto
        self.smtp_username = config.smtp_username  # Definir en 'config' o usar una variable por defecto
        self.smtp_password = config.smtp_password  # Definir en 'config' o usar una variable por defecto

        self.imap_server = config.imap_server  # Definir en 'config' o usar una variable por defecto
        self.imap_port = config.imap_port  # Definir en 'config' o usar una variable por defecto
        self.imap_username = config.imap_username  # Definir en 'config' o usar una variable por defecto
        self.imap_password = config.imap_password  # Definir en 'config' o usar una variable por defecto


    def send_email(self, subject, body, to_address_list):
        from_address = self.smtp_username

        # mensaje MIME
        msg = MIMEMultipart()
        msg['From'] = from_address
        msg['To'] = ", ".join(to_address_list)  # Unir la lista de destinatarios con comas
        msg['Subject'] = subject

        # cuerpo del mensaje, codificación UTF-8
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as smtp:
            smtp.login(self.smtp_username, self.smtp_password)
            smtp.sendmail(from_address, to_address_list, msg.as_string())


    # def get_email(self):
    #     # conexión IMAP
    #     with imaplib.IMAP4_SSL(self.imap_server, self.imap_port) as mail:
    #         try:
    #             mail.login(self.imap_username, self.imap_password)
    #             mail.select("inbox")
    #             # Buscar correos
    #             status, messages = mail.search(None, 'UNSEEN')
    #             # Lista de ids
    #             mail_ids = messages[0].split()

    #             for mail_id in mail_ids:
    #                 status, data = mail.fetch(mail_id, '(RFC822)')

    #                 for response_part in data:
    #                     if isinstance(response_part, tuple):
    #                         msg = email.message_from_bytes(response_part[1])
    #                         subject, encoding = decode_header(msg["Subject"])[0]

    #                         # Decodificar el asunto si está en bytes
    #                         if isinstance(subject, bytes):
    #                             subject = subject.decode(encoding if encoding else "utf-8")

    #                         # Obtener remitente
    #                         from_ = msg.get("From")

    #                         print(f"Correo de: {from_}")
    #                         print(f"Asunto: {subject}")

    #                         # Si el correo tiene varias partes
    #                         if msg.is_multipart():
    #                             for part in msg.walk():
    #                                 if part.get_content_type() == "text/plain":
    #                                     body = part.get_payload(decode=True).decode()
    #                                     print(f"Cuerpo: {body}")
    #                         else:
    #                             body = msg.get_payload(decode=True).decode()
    #                             print(f"Cuerpo: {body}")

    #         except Exception as e:
    #             print(f"Error al obtener correos: {e}")

    def get_unread_emails(self):
        """
        Recupera todos los correos electrónicos no leídos y los marca como leídos.

        Returns:
            List[dict]: Lista de diccionarios con la información de cada correo no leído.
        """
        # Lista para almacenar la información de los correos
        emails_info = []

        # Conexión IMAP
        with imaplib.IMAP4_SSL(self.imap_server, self.imap_port) as mail:
            try:
                mail.login(self.imap_username, self.imap_password)
                mail.select("inbox")

                # Buscar correos no leídos (UNSEEN)
                status, messages = mail.search(None, 'UNSEEN')
                mail_ids = messages[0].split()

                for mail_id in mail_ids:
                    # Fetch el correo por ID
                    status, data = mail.fetch(mail_id, '(RFC822)')
                    
                    for response_part in data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])
                            
                            # Obtener fecha del mensaje
                            email_date = email.utils.parsedate_to_datetime(msg["Date"])
                            
                            # Decodificar el asunto
                            subject, encoding = decode_header(msg["Subject"])[0]
                            if isinstance(subject, bytes):
                                subject = subject.decode(encoding if encoding else "utf-8")
                            
                            # Obtener remitente
                            from_ = msg.get("From")
                            
                            # Obtener cuerpo del mensaje
                            body = ""
                            if msg.is_multipart():
                                for part in msg.walk():
                                    if part.get_content_type() == "text/plain" and not part.get("Content-Disposition"):
                                        body = part.get_payload(decode=True).decode()
                                        break
                            else:
                                body = msg.get_payload(decode=True).decode()
                            
                            # Guardar la información en un diccionario
                            email_info = {
                                "from": from_,
                                "date": email_date,
                                "subject": subject,
                                "body": body
                            }
                            
                            emails_info.append(email_info)

                            # Marcar el correo como leído (SEEN)
                            mail.store(mail_id, '+FLAGS', '\\Seen')

            except Exception as e:
                print(f"Error al obtener correos: {e}")
            
        return emails_info

