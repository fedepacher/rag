import os
import smtplib
import imaplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.header import decode_header
from config import Config
from logger_config import logger


class EmailManager:
    def __init__(self):
        config = Config()
        logger.info("Loading EmailManager configuration")
        self.db_name = config.db_name
        self.db_user = config.db_user
        self.db_pass = config.db_pass
        self.db_host = config.db_host
        self.db_port = config.db_port
        self.access_token_expire_minutes = config.access_token_expire_minutes
        self.secret_key = config.secret_key

        self.mongo_host = config.mongo_host
        self.mongo_db_name = config.mongo_db_name
        self.mongo_port = config.mongo_port
        self.mongo_user = config.mongo_user
        self.mongo_pass = config.mongo_pass

        self.smtp_server = config.smtp_server
        self.smtp_port = config.smtp_port
        self.smtp_username = config.smtp_username
        self.smtp_password = config.smtp_password

        self.imap_server = config.imap_server
        self.imap_port = config.imap_port
        self.imap_username = config.imap_username
        self.imap_password = config.imap_password
        
        self.email_rest_sec = config.email_rest_sec

        self.logo_path = config.logo_path
        self.email_tmpt_path = config.email_tmpt_path


    def send_email(self, subject, consulta, body, to_address_list) -> bool:
        """
        Intenta mandar el email. Si lo consigue retorna True. Caso contrario False
        """
        try:
            logger.info(f"Sending email to {to_address_list}")
            from_address = self.smtp_username

            # mensaje MIME
            msg = MIMEMultipart("related") # "related" -> p agregar logo
            msg['From'] = from_address
            msg['To'] = ", ".join(to_address_list)  # Unir la lista de destinatarios con comas
            msg['Subject'] = subject

            # plantilla HTML p email
            with open(self.email_tmpt_path, 'r', encoding='utf-8') as file:
                html_template = file.read()

            # Reemplazar las variables en la plantilla
            html_content = (html_template
                            .replace('{{username}}', "estimado usuario")
                            .replace('{{pregunta}}', consulta)
                            .replace('{{respuesta}}', body))

            msg.attach(MIMEText(html_content, 'html', 'utf-8'))

            
            # intento adjuntar logo (pero si falla igual sigue)
            try: 
                with open(self.logo_path, 'rb') as img_file:
                    img = MIMEImage(img_file.read())
                    img.add_header('Content-ID', '<logo_image>')
                    img.add_header('Content-Disposition', 'inline', filename=os.path.basename(self.logo_path))
                    msg.attach(img)
            except Exception as e:
                logger.error(f"Error agregando el logo: {e}")

            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as smtp:
                smtp.login(self.smtp_username, self.smtp_password)
                smtp.sendmail(from_address, to_address_list, msg.as_string())
            return True
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False


    def get_unread_emails(self):
        """
        Recupera todos los correos electrónicos no leídos y los marca como leídos.

        Returns:
            List[dict]: Lista de diccionarios con la información de cada correo no leído.
        """
        # Lista para almacenar la información de los correos
        emails_info = []

        # Conexión IMAP
        
        try:
            logger.info(f"Getting emails from {self.imap_server}")
            with imaplib.IMAP4_SSL(self.imap_server, self.imap_port) as mail:
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
            logger.error(f"Error getting emails: {e}")
            return []

        logger.info(f"Unread emails got: {len(emails_info)}")
        return emails_info

