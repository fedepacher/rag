import os
import logging

class Config:
    def __init__(self):
        # Cargar configuración
            logging.info(f"Loading configuration.")
            self.db_name = os.getenv("DB_NAME", "test_sql")
            self.db_user = os.getenv("DB_USER", "rag-system")
            self.db_pass = os.getenv("DB_PASS", "rag-system")
            self.db_host = os.getenv("DB_HOST", "database")
            self.db_port = os.getenv("DB_PORT", 3306)
            self.access_token_expire_minutes = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440)
            self.secret_key = os.getenv("SECRET_KEY", "6b3d75e968f26f3be3443503efefd761e22d5e173fea95646a3659e673ebb97b")

            self.mongo_host = os.getenv("MONGO_HOST", "mongodb")
            self.mongo_db_name = os.getenv("MONGO_DB_NAME", "test_nosql")
            self.mongo_port = os.getenv("MONGO_PORT", 27017)
            self.mongo_user = os.getenv("MONGO_USER", "mongoadmin")
            self.mongo_pass = os.getenv("MONGO_PASS", "secret")

            self.smtp_server = os.getenv("SMTP_SERVER", "smtp.zoho.com")
            self.smtp_port = int(os.getenv("SMTP_PORT", 465))
            self.smtp_username = os.getenv("SMTP_USERNAME", "tambo2022@zohomail.com")
            # self.smtp_password = os.getenv("SMTP_PASSWORD")

            self.imap_server = os.getenv("IMAP_SERVER", "imap.zoho.com")
            self.imap_port = int(os.getenv("IMAP_PORT", 993))
            self.imap_username = os.getenv("IMAP_USERNAME", "tambo2022@zohomail.com")
            # self.imap_password = os.getenv("IMAP_PASSWORD")

            self.email_rest_sec = os.getenv("EMAIL_REST_SEC")
            self.email_sbjt_code = os.getenv("EMAIL_SBJT_CODE")

            self.logo_path = os.getenv("LOGO_PATH")
            self.email_tmpt_path = os.getenv("EMAIL_TMPT_PATH")
